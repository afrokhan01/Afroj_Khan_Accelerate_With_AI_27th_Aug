from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MappingMemoryRecord:
	source_signature: str
	mapping: dict[str, str]
	confidence: float


class AgentMemory:
	def __init__(self, memory_file: Path) -> None:
		self.memory_file = memory_file
		self.memory_file.parent.mkdir(parents=True, exist_ok=True)
		if not self.memory_file.exists():
			self.memory_file.write_text("[]", encoding="utf-8")

	def load_records(self) -> list[MappingMemoryRecord]:
		payload: list[dict[str, Any]] = json.loads(self.memory_file.read_text(encoding="utf-8"))
		return [MappingMemoryRecord(**item) for item in payload]

	def find_mapping(self, source_signature: str) -> MappingMemoryRecord | None:
		for record in self.load_records():
			if record.source_signature == source_signature:
				return record
		return None

	def remember(self, record: MappingMemoryRecord) -> None:
		records = self.load_records()
		records = [r for r in records if r.source_signature != record.source_signature]
		records.append(record)
		self.memory_file.write_text(
			json.dumps([r.__dict__ for r in records], indent=2),
			encoding="utf-8",
		)


def get_chroma_client() -> Any:
	import chromadb
	from core.config import CHROMA_DIR

	return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = "idamp_memory") -> Any:
	client = get_chroma_client()
	return client.get_or_create_collection(name=name)


def store_document(doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
	try:
		collection = get_collection()
		collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata or {}])
	except Exception:
		pass


def query_memory(query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
	try:
		collection = get_collection()
		result = collection.get(limit=n_results)
		ids = result.get("ids", [])
		documents = result.get("documents", [])
		metadatas = result.get("metadatas", [])
		return [
			{"id": ids[i], "document": documents[i], "metadata": metadatas[i]}
			for i in range(len(ids))
		]
	except Exception:
		return []
