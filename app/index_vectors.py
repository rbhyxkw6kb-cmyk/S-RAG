import json

from app.core.config import settings
from app.services.instruction_library import InstructionLibrary


if __name__ == "__main__":
    library = InstructionLibrary(settings.knowledge_db)
    print(json.dumps(library.backfill_vectors(), ensure_ascii=False))
