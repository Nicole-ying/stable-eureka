from __future__ import annotations

import eg_rsa.run_single_chain_iterative as base
from eg_rsa.single_chain.controller_v2 import SearchController
from eg_rsa.single_chain.expert_memory_v2 import build_expert_memory_context
from eg_rsa.single_chain.memory_manager_v2 import MemoryManager


# Keep the original iterative runner intact, but swap in audit-aware
# context, memory, and controller implementations for this entrypoint.
base.SearchController = SearchController
base.build_expert_memory_context = build_expert_memory_context
base.MemoryManager = MemoryManager

run_iterative = base.run_iterative
main = base.main


if __name__ == "__main__":
    main()
