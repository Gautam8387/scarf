"""CELLxGENE registration and self-contained DuckDB publication."""

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID

import duckdb
import httpx
from huggingface_hub import BucketFolder, list_bucket_tree
from huggingface_hub.errors import EntryNotFoundError
from natsort import natsorted

from .._storage import Bucket, dataset_prefix, retry
from .models import DatasetRecord, DatasetVersion, FacetTerm

_CURATION_API = "https://api.cellxgene.cziscience.com/curation/v1"


@dataclass(frozen=True)
class SourceMetadata:
    collection: dict[str, Any]
    dataset: dict[str, Any]
    source_url: str
    source_bytes: int | None


def fetch_collection(collection_id: str) -> tuple[bytes, dict[str, Any]]:
    """Fetch the current published collection and retain its response bytes."""
    response = _get(f"{_CURATION_API}/collections/{collection_id}")
    collection = response.json()
    if not isinstance(collection, dict):
        raise ValueError("CELLxGENE collection response must be a JSON object")
    if collection.get("collection_id") != collection_id:
        raise ValueError("CELLxGENE returned a different collection identity")
    return response.content, collection


def _get(url: str) -> httpx.Response:
    def request():
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()
        return response

    return retry(request)


def list_collection_ids() -> list[str]:
    """List public CELLxGENE collection IDs without registering or downloading."""
    rows = _get(f"{_CURATION_API}/collections?visibility=PUBLIC").json()
    if not isinstance(rows, list):
        raise ValueError("CELLxGENE collection list must be an array")
    return sorted({str(UUID(row["collection_id"])) for row in rows})


def source_metadata(
    collection: dict[str, Any],
    dataset: dict[str, Any],
) -> SourceMetadata:
    """Validate one dataset's H5AD asset without refetching its collection."""
    if not dataset.get("dataset_id") or not dataset.get("dataset_version_id"):
        raise ValueError("CELLxGENE dataset is missing its stable or version ID")
    assets = [
        asset
        for asset in dataset.get("assets", [])
        if str(asset.get("filetype", "")).upper() == "H5AD"
    ]
    if len(assets) != 1 or not assets[0].get("url"):
        raise ValueError("CELLxGENE dataset must provide one H5AD download asset")
    asset = assets[0]
    size = asset.get("filesize")
    return SourceMetadata(
        collection=collection,
        dataset=dataset,
        source_url=asset["url"],
        source_bytes=int(size) if size is not None else None,
    )


def _publication_citation(collection: dict[str, Any]) -> str:
    publisher = collection.get("publisher_metadata") or {}
    authors = publisher.get("authors") or []
    first_author = (
        authors[0].get("family") or authors[0].get("name", "") if authors else ""
    )
    author = f"{first_author} et al." if len(authors) > 1 else first_author
    year = publisher.get("published_year")
    return " ".join(
        str(part)
        for part in (author, f"({year})" if year else "", publisher.get("journal"))
        if part
    )


def metadata_manifest_fields(metadata: SourceMetadata) -> dict[str, Any]:
    """Return available manifest fields from the unmodified API records."""
    collection = metadata.collection
    dataset = metadata.dataset
    organism = ", ".join(
        item["label"] for item in (dataset.get("organism") or []) if item.get("label")
    )
    fields = {
        "title": dataset.get("title") or collection.get("name"),
        "citation": dataset.get("citation") or _publication_citation(collection),
        "doi": collection.get("doi"),
        "schemaVersion": dataset.get("schema_version"),
        "organism": organism,
        "metadataSource": "curation_api",
    }
    return {key: value for key, value in fields.items() if value}


_ONTOLOGY_FACETS = (
    "organism",
    "assay",
    "tissue",
    "disease",
    "cell_type",
    "sex",
    "development_stage",
)


def is_main_dataset(dataset: dict) -> bool:
    """Keep any-primary datasets; reject missing or contradictory primary metadata."""
    dataset_id = dataset.get("dataset_id", "unknown")
    count = dataset.get("primary_cell_count")
    cells = dataset.get("cell_count")
    flags = dataset.get("is_primary_data")
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError(f"Invalid primary_cell_count for dataset {dataset_id}")
    if cells is not None and (type(cells) is not int or cells < 0):
        raise ValueError(f"Invalid cell_count for dataset {dataset_id}")
    if flags is not None and (
        not isinstance(flags, list) or any(type(flag) is not bool for flag in flags)
    ):
        raise ValueError(f"Invalid is_primary_data list for dataset {dataset_id}")
    primary_flags = set(flags or [])
    if count is None and not primary_flags:
        raise ValueError(f"Primary-data metadata is missing for dataset {dataset_id}")
    contradictory = count is not None and (
        (cells is not None and count > cells)
        or (bool(primary_flags) and (count > 0) != (True in primary_flags))
        or (cells is not None and primary_flags == {True} and count != cells)
        or (cells is not None and primary_flags == {True, False} and count == cells)
    )
    if contradictory:
        raise ValueError(
            f"Primary-data metadata contradicts itself for dataset {dataset_id}"
        )
    return count > 0 if count is not None else True in primary_flags


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")[:24].rstrip("_")


def _publication(collection: dict) -> tuple[str | None, int | None, str]:
    publisher = collection.get("publisher_metadata") or {}
    authors = publisher.get("authors") or []
    first = authors[0] if authors else {}
    author = first.get("family") or first.get("name") or None
    if first.get("family"):
        author_slug = _slug(first["family"].replace("-", ""))
    else:
        words = re.findall(r"\w+", first.get("name", ""), flags=re.UNICODE)
        while words and words[0].lower() in {"the", "a", "an"}:
            words.pop(0)
        author_slug = _slug(words[0]) if words else "unpub"
    year = publisher.get("published_year")
    return author, int(year) if year is not None else None, author_slug or "unpub"


def _facets(dataset: dict) -> dict[str, list[FacetTerm]]:
    facets = {}
    for facet in _ONTOLOGY_FACETS:
        terms = {
            (item.get("ontology_term_id"), item["label"])
            for item in dataset.get(facet) or []
        }
        facets[facet] = [
            FacetTerm(termId=term_id, label=label)
            for term_id, label in sorted(
                terms, key=lambda term: (term[1], term[0] or "")
            )
        ]
    facets["organ"] = []
    facets["suspension_type"] = [
        FacetTerm(label=label)
        for label in sorted(set(dataset.get("suspension_type") or []))
    ]
    return facets


def _name_parts(collection: dict, facets: dict[str, list[FacetTerm]]) -> list[str]:
    _, year, author = _publication(collection)
    parts = [author, f"{year:04d}" if year is not None else "0000"]
    organisms = sorted({term.label for term in facets["organism"]})
    if organisms != ["Homo sapiens"]:
        organism = (
            {"Mus musculus": "mouse"}.get(organisms[0], organisms[0])
            if len(organisms) == 1
            else "multispecies"
            if organisms
            else "unknown"
        )
        parts.append(_slug(organism))
    tissues = {term.label for term in facets["tissue"]}
    parts.append(_slug(next(iter(tissues))) if len(tissues) == 1 else "multitissue")
    diseases = {term.label for term in facets["disease"]}
    non_normal = diseases - {"normal"}
    disease = (
        "healthy"
        if diseases == {"normal"}
        else next(iter(non_normal))
        if len(non_normal) == 1
        else "multidisease"
    )
    parts.append(_slug(disease))
    return parts


def _name_with_suffix(parts: list[str], suffix: str) -> str:
    shortened = list(parts)
    while len("_".join([*shortened, suffix])) > 80:
        longest = max(range(len(shortened)), key=lambda index: len(shortened[index]))
        shortened[longest] = shortened[longest][:-1].rstrip("_")
    return "_".join([*shortened, suffix])


def _assign_name(parts: list[str], dataset_id: UUID, reserved: dict[str, UUID]) -> str:
    for length in (8, 12, 32):
        name = _name_with_suffix(parts, dataset_id.hex[:length])
        if name not in reserved or reserved[name] == dataset_id:
            reserved[name] = dataset_id
            return name
    raise ValueError(
        f"Could not assign a unique Cytebase name for dataset {dataset_id}"
    )


def _readme(record: DatasetRecord, collection: dict) -> bytes:
    lines = [f"# {record.title or record.cytebaseId}", ""]
    if record.citation:
        lines.extend([record.citation, ""])
    if record.doi:
        lines.extend([f"DOI: https://doi.org/{record.doi}", ""])
    lines.extend(
        [
            f"Collection: {record.cellxgeneUrl}",
            f"Dataset ID: `{record.datasetId}`",
            f"Latest dataset version: `{record.latestVersionId}`",
            f"Source H5AD: {record.sourceUrl}",
        ]
    )
    if record.explorerUrl:
        lines.append(f"CELLxGENE Explorer: {record.explorerUrl}")
    lines.extend(["", "## Metadata", ""])
    for facet, terms in record.facets.items():
        if terms:
            lines.append(f"- {facet}: {', '.join(term.label for term in terms)}")
    if collection.get("description"):
        lines.extend(["", "## Abstract", "", collection["description"]])
    return ("\n".join(lines) + "\n").encode()


def prepare_registration(
    collections: list[tuple[bytes, dict]],
    existing: list[DatasetRecord],
    *,
    pipeline_version: str,
    now: datetime | None = None,
) -> tuple[list[DatasetRecord], list[dict], list[tuple[bytes, str]]]:
    """Prepare all main records, collection rows, and raw-metadata uploads atomically."""
    timestamp = now or datetime.now(UTC)
    by_id = {record.datasetId: record for record in existing}
    reserved = {record.cytebaseId: record.datasetId for record in existing}
    if len(by_id) != len(existing) or len(reserved) != len(existing):
        raise ValueError("Existing registrations contain duplicate IDs or names")
    owners = {record.datasetId: record.collectionId for record in existing}
    seen_collections = set()
    pending = []
    collection_rows = []
    for raw, collection in sorted(
        collections, key=lambda item: item[1]["collection_id"]
    ):
        collection_id = UUID(collection["collection_id"])
        if collection_id in seen_collections:
            raise ValueError(f"Collection was supplied more than once: {collection_id}")
        seen_collections.add(collection_id)
        datasets = collection.get("datasets")
        if not isinstance(datasets, list):
            raise ValueError(f"Collection {collection_id} has no dataset list")
        skipped = {}
        seen_datasets = set()
        main_count = 0
        for dataset in sorted(datasets, key=lambda item: item["dataset_id"]):
            dataset_id = UUID(dataset["dataset_id"])
            if dataset_id in seen_datasets:
                raise ValueError(f"Collection repeats dataset {dataset_id}")
            seen_datasets.add(dataset_id)
            if dataset_id in owners and owners[dataset_id] != collection_id:
                raise ValueError(f"Dataset {dataset_id} occurs in multiple collections")
            owners[dataset_id] = collection_id
            if is_main_dataset(dataset):
                pending.append((raw, collection, dataset))
                main_count += 1
            else:
                if dataset_id in by_id:
                    raise ValueError(
                        f"Registered dataset {dataset_id} is now all-secondary; "
                        "review its existing files before removing its registration"
                    )
                skipped[str(dataset_id)] = "All cells are secondary (no primary cells)"
        author, year, _ = _publication(collection)
        registered = [
            record.registeredAt
            for record in existing
            if record.collectionId == collection_id
        ]
        collection_rows.append(
            {
                "collection_id": str(collection_id),
                "name": collection.get("name"),
                "description": collection.get("description"),
                "doi": collection.get("doi"),
                "first_author": author,
                "year": year,
                "journal": (collection.get("publisher_metadata") or {}).get("journal"),
                "consortia": collection.get("consortia") or [],
                "n_datasets_total": len(datasets),
                "n_datasets_main": main_count,
                "skipped_dataset_ids": list(skipped),
                "skipped_dataset_reasons": skipped,
                "registered_at": min(registered) if registered else timestamp,
            }
        )

    records = []
    uploads = []
    for raw, collection, dataset in pending:
        source = source_metadata(collection, dataset)
        dataset_id = UUID(dataset["dataset_id"])
        version_id = UUID(dataset["dataset_version_id"])
        previous = by_id.get(dataset_id)
        facets = _facets(dataset)
        name = (
            previous.cytebaseId
            if previous
            else _assign_name(_name_parts(collection, facets), dataset_id, reserved)
        )
        versions = list(previous.versions) if previous else []
        if not any(version.datasetVersionId == version_id for version in versions):
            versions.append(
                DatasetVersion(datasetVersionId=version_id, seenAt=timestamp)
            )
        status = previous.status if previous else "registered"
        if previous and previous.latestVersionId != version_id:
            status = "update_available" if previous.processedVersionId else "registered"
        author, year, _ = _publication(collection)
        manifest_fields = metadata_manifest_fields(source)
        record = DatasetRecord(
            **(
                (previous.model_dump() if previous else {})
                | {
                    "cytebaseId": name,
                    "datasetId": dataset_id,
                    "collectionId": UUID(collection["collection_id"]),
                    "latestVersionId": version_id,
                    "versions": versions,
                    "title": manifest_fields.get("title"),
                    "citation": manifest_fields.get("citation"),
                    "doi": collection.get("doi"),
                    "firstAuthor": author,
                    "year": year,
                    "facets": facets,
                    "cellCount": dataset.get("cell_count"),
                    "primaryCellCount": dataset.get("primary_cell_count"),
                    "nGenes": dataset.get("feature_count"),
                    "schemaVersion": dataset.get("schema_version"),
                    "status": status,
                    "sourceUrl": source.source_url,
                    "sourceBytes": source.source_bytes,
                    "cellxgeneUrl": collection.get("collection_url")
                    or f"https://cellxgene.cziscience.com/collections/{collection['collection_id']}",
                    "explorerUrl": dataset.get("explorer_url"),
                    "registeredAt": previous.registeredAt if previous else timestamp,
                    "updatedAt": timestamp,
                    "pipelineVersion": pipeline_version,
                }
            )
        )
        # Saved URIs describe published objects. Registration must not relocate
        # an existing store; only completed processing publishes new locations.
        records.append(record)
        prefix = f"datasets/{name}"
        uploads.extend(
            [
                (raw, f"{prefix}/cellxgene/collection.json"),
                (
                    json.dumps(dataset, indent=2, ensure_ascii=False).encode(),
                    f"{prefix}/cellxgene/dataset.json",
                ),
                (_readme(record, collection), f"{prefix}/README.md"),
            ]
        )
    return records, collection_rows, uploads


FACETS = (
    "tissue",
    "organ",
    "disease",
    "assay",
    "organism",
    "cell_type",
    "sex",
    "development_stage",
    "suspension_type",
)
_FACET_COLUMNS = {
    facet: f"{facet}_labels" if facet != "suspension_type" else "suspension_types"
    for facet in FACETS
}
_FACETS_WITH_IDS = ("organism", "assay", "tissue", "organ", "disease", "cell_type")
_SCHEMAS = {
    "datasets": [
        ("cytebase_id", "VARCHAR PRIMARY KEY"),
        ("dataset_id", "VARCHAR"),
        ("collection_id", "VARCHAR"),
        ("latest_version_id", "VARCHAR"),
        ("processed_version_id", "VARCHAR"),
        ("title", "VARCHAR"),
        ("citation", "VARCHAR"),
        ("doi", "VARCHAR"),
        ("first_author", "VARCHAR"),
        ("year", "INTEGER"),
        ("organism_labels", "VARCHAR[]"),
        ("organism_ids", "VARCHAR[]"),
        ("assay_labels", "VARCHAR[]"),
        ("assay_ids", "VARCHAR[]"),
        ("tissue_labels", "VARCHAR[]"),
        ("tissue_ids", "VARCHAR[]"),
        ("organ_labels", "VARCHAR[]"),
        ("organ_ids", "VARCHAR[]"),
        ("disease_labels", "VARCHAR[]"),
        ("disease_ids", "VARCHAR[]"),
        ("cell_type_labels", "VARCHAR[]"),
        ("cell_type_ids", "VARCHAR[]"),
        ("sex_labels", "VARCHAR[]"),
        ("development_stage_labels", "VARCHAR[]"),
        ("suspension_types", "VARCHAR[]"),
        ("cell_count", "BIGINT"),
        ("primary_cell_count", "BIGINT"),
        ("n_genes", "INTEGER"),
        ("schema_version", "VARCHAR"),
        ("status", "VARCHAR"),
        ("zarr_uri", "VARCHAR"),
        ("h5ad_uri", "VARCHAR"),
        ("cellxgene_url", "VARCHAR"),
        ("explorer_url", "VARCHAR"),
        ("registered_at", "TIMESTAMPTZ"),
        ("processed_at", "TIMESTAMPTZ"),
        ("pipeline_version", "VARCHAR"),
    ],
    "collections": [
        ("collection_id", "VARCHAR PRIMARY KEY"),
        ("name", "VARCHAR"),
        ("description", "VARCHAR"),
        ("doi", "VARCHAR"),
        ("first_author", "VARCHAR"),
        ("year", "INTEGER"),
        ("journal", "VARCHAR"),
        ("consortia", "VARCHAR[]"),
        ("n_datasets_total", "BIGINT"),
        ("n_datasets_main", "BIGINT"),
        ("skipped_dataset_ids", "VARCHAR[]"),
        ("skipped_dataset_reasons", "MAP(VARCHAR, VARCHAR)"),
        ("registered_at", "TIMESTAMPTZ"),
    ],
    "dataset_terms": [
        ("cytebase_id", "VARCHAR"),
        ("facet", "VARCHAR"),
        ("term_id", "VARCHAR"),
        ("label", "VARCHAR"),
        ("label_rank", "BIGINT"),
    ],
}


def _dataset_row(record: DatasetRecord) -> dict[str, Any]:
    data = record.model_dump(mode="json")
    row = {}
    for column, _ in _SCHEMAS["datasets"]:
        first, *rest = column.split("_")
        key = first + "".join(part.capitalize() for part in rest)
        row[column] = data.get(key)
    for facet, column in _FACET_COLUMNS.items():
        terms = record.facets.get(facet, [])
        row[column] = list(dict.fromkeys(term.label for term in terms))
        if facet in _FACETS_WITH_IDS:
            row[f"{facet}_ids"] = list(
                dict.fromkeys(term.termId for term in terms if term.termId is not None)
            )
    return row


def _term_rows(records: list[DatasetRecord]) -> list[dict[str, Any]]:
    ranks = {}
    for facet in FACETS:
        labels = {
            term.label for record in records for term in record.facets.get(facet, [])
        }
        ranks[facet] = {
            label: rank for rank, label in enumerate(natsorted(sorted(labels)), start=1)
        }
    rows = []
    for record in sorted(records, key=lambda item: item.cytebaseId):
        for facet in FACETS:
            terms = {(term.termId, term.label) for term in record.facets.get(facet, [])}
            for term_id, label in sorted(
                terms, key=lambda term: (ranks[facet][term[1]], term[0] or "")
            ):
                rows.append(
                    {
                        "cytebase_id": record.cytebaseId,
                        "facet": facet,
                        "term_id": term_id,
                        "label": label,
                        "label_rank": ranks[facet][label],
                    }
                )
    return rows


def _create_table(connection: duckdb.DuckDBPyConnection, name: str) -> None:
    fields = ", ".join(f"{column} {dtype}" for column, dtype in _SCHEMAS[name])
    connection.execute(f"CREATE TABLE {name} ({fields})")


def _write_catalog(rows: dict[str, list[dict]], directory: Path) -> dict[str, Path]:
    """Materialize every table; the resulting database has no external dependencies."""
    database_path = directory / "cytebase.duckdb"
    if database_path.exists():
        raise FileExistsError(f"Catalog database already exists: {database_path}")
    with duckdb.connect(str(database_path)) as database:
        for name, schema in _SCHEMAS.items():
            _create_table(database, name)
            values = [
                [
                    row.get(column, [] if dtype.endswith("[]") else {})
                    if dtype.endswith("[]") or dtype.startswith("MAP(")
                    else row.get(column)
                    for column, dtype in schema
                ]
                for row in rows[name]
            ]
            if values:
                placeholders = ", ".join("?" for _ in schema)
                database.executemany(
                    f"INSERT INTO {name} VALUES ({placeholders})", values
                )
        database.execute("CHECKPOINT")
    with database_path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    hash_path = directory / "cytebase.duckdb.sha256"
    hash_path.write_text(f"{checksum}  cytebase.duckdb\n", encoding="ascii")
    return {"cytebase.duckdb": database_path, "cytebase.duckdb.sha256": hash_path}


def _materialize(result: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [column[0] for column in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def load_record(storage: Bucket, key: str) -> DatasetRecord:
    raw = storage.read_json(f"{dataset_prefix(key)}/dataset.json")
    if raw is None:
        raise FileNotFoundError(f"Dataset {key} is not registered")
    record = DatasetRecord.model_validate(raw)
    if record.cytebaseId != key:
        raise ValueError("Dataset identity does not match its directory")
    return record


def list_records(storage: Bucket) -> list[DatasetRecord]:
    """Read dataset records using a shallow listing, never traversing Zarr files."""

    def folders():
        try:
            return list(
                list_bucket_tree(
                    storage.bucket_id,
                    prefix="datasets/",
                    recursive=False,
                    token=storage.token,
                )
            )
        except EntryNotFoundError:
            return []

    keys = {
        item.path.removeprefix("datasets/").rstrip("/")
        for item in retry(folders)
        if isinstance(item, BucketFolder)
    }
    records = [load_record(storage, key) for key in sorted(keys)]
    if len({record.datasetId for record in records}) != len(records):
        raise ValueError(
            "Multiple registered directories use the same CELLxGENE dataset ID"
        )
    return records


def select_dataset_ids(request, storage: Bucket) -> list[str]:
    if request.cytebaseIds is not None:
        for key in request.cytebaseIds:
            dataset_prefix(key)
        return list(dict.fromkeys(request.cytebaseIds))
    selected = set(request.collectionIds or [request.collectionId])
    return [
        record.cytebaseId
        for record in list_records(storage)
        if record.collectionId in selected
    ]


def publish_catalog(
    storage: Bucket, updates: list[DatasetRecord], collections: list[dict]
) -> dict:
    """Merge changed rows into a self-contained snapshot, preserving other collections."""
    with TemporaryDirectory(prefix="cytebase-catalog-") as directory:
        root = Path(directory)
        previous = root / "previous.duckdb"
        rows = {name: [] for name in _SCHEMAS}
        try:
            storage.download("catalog/cytebase.duckdb", previous)
        except EntryNotFoundError:
            # First publication reconstructs existing registered datasets too.
            known = {record.cytebaseId: record for record in list_records(storage)}
            known.update({record.cytebaseId: record for record in updates})
            updates = list(known.values())
        else:
            with duckdb.connect(str(previous), read_only=True) as con:
                rows = {
                    name: _materialize(con.execute(f"SELECT * FROM {name}"))
                    for name in _SCHEMAS
                }
        datasets = {row["cytebase_id"]: row for row in rows["datasets"]}
        datasets.update({record.cytebaseId: _dataset_row(record) for record in updates})
        collection_rows = {row["collection_id"]: row for row in rows["collections"]}
        for row in collections:
            old = collection_rows.get(row["collection_id"])
            if old:
                row = row | {"registered_at": old["registered_at"]}
            collection_rows[row["collection_id"]] = row
        changed = {record.cytebaseId for record in updates}
        terms = [
            row for row in rows["dataset_terms"] if row["cytebase_id"] not in changed
        ] + _term_rows(updates)
        ranks = {
            facet: {
                label: i
                for i, label in enumerate(
                    natsorted(
                        sorted({row["label"] for row in terms if row["facet"] == facet})
                    ),
                    1,
                )
            }
            for facet in FACETS
        }
        for row in terms:
            row["label_rank"] = ranks[row["facet"]][row["label"]]
        paths = _write_catalog(
            {
                "datasets": list(datasets.values()),
                "collections": list(collection_rows.values()),
                "dataset_terms": terms,
            },
            root,
        )
        storage.upload([(paths["cytebase.duckdb"], "catalog/cytebase.duckdb")])
        storage.upload(
            [(paths["cytebase.duckdb.sha256"], "catalog/cytebase.duckdb.sha256")]
        )
        return {
            "status": "done",
            "datasets": len(datasets),
            "collections": len(collection_rows),
            "catalogUri": f"{storage.root}/catalog/cytebase.duckdb",
            "catalogSha256": paths["cytebase.duckdb.sha256"].read_text().split()[0],
        }


def run_catalog(request: dict, storage: Bucket, assert_owner) -> dict:
    """Register requested collections or publish the supplied dataset changes."""
    collection_rows = []
    records = []
    if request.get("collectionIds"):
        ids = list(
            dict.fromkeys(str(UUID(value)) for value in request["collectionIds"])
        )
        records, collection_rows, files = prepare_registration(
            [fetch_collection(key) for key in ids],
            list_records(storage),
            pipeline_version=os.environ["CYTEBASE_PIPELINE_VERSION"],
        )
        assert_owner()
        if files:
            storage.upload(files)
        for record in records:
            assert_owner()
            storage.write_json(
                f"{dataset_prefix(record.cytebaseId)}/dataset.json",
                record.model_dump(mode="json"),
            )
    elif "updates" in request:
        records = [DatasetRecord.model_validate(row) for row in request["updates"]]
    else:
        records = list_records(storage)
    assert_owner()
    result = publish_catalog(storage, records, collection_rows)
    if request.get("collectionIds"):
        result.update(
            registeredDatasets=[
                {"cytebaseId": record.cytebaseId, "status": record.status}
                for record in records
            ],
            registeredCollections=collection_rows,
        )
    return result
