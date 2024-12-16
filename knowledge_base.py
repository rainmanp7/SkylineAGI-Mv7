# Beginning of knowledge_base.py
from collections import deque
from threading import Lock
from typing import List, Dict, Any, Tuple, Optional

# this number arrangement here is for complexity placement. Match the nearest and put it in the right category dataset. The information here is for the assimilation to follow and just placed here for import etc.. and how the knowledge base aka domain dataset.

# each domain dataset has this num arrangement.

from internal_process_monitor import InternalProcessMonitor

# Can't use leading 000's in Python.

class TieredKnowledgeBase:
    # Define complexity tiers
    TIERS = {
    # 1st Section
    'easy': (1111, 1389),
    'simp': (1390, 1668),
    'norm': (1669, 1947),
    # 2nd Section
    'mods': (1948, 2226),
    'hard': (2227, 2505),
    'para': (2506, 2784),
    # 3rd Section
    'vice': (2785, 3063),
    'zeta': (3064, 3342),
    'tetr': (3343, 3621),
    # 4th Section
    'eafv': (3622, 3900),
    'sipo': (3901, 4179),
    'nxxm': (4180, 4458),
    # 5th Section
    'mids': (4459, 4737),
    'haod': (4738, 5016),
    'parz': (5017, 5295),
    # 6th Section
    'viff': (5296, 5574),
    'zexa': (5575, 5853),
    'sip8': (5854, 6132),
    # 7th Section
    'nxVm': (6133, 6411),
    'Vids': (6412, 6690),
    'ha3d': (6691, 6969),
    # 8th Section
    'pfgz': (6970, 7248),
    'vpff': (7249, 7527),
    'z9xa': (7528, 7806),
    # 9th Section
    'Tipo': (7807, 8085),
    'nxNm': (8086, 8364),
    'mPd7': (8365, 9918)
}


    def __init__(self, max_recent_items: int = 100):
        self.knowledge_bases = {tier: {} for tier in self.TIERS.keys()}
        self.recent_updates = deque(maxlen=max_recent_items)
        self.lock = Lock()
        self.knowledge_base_monitor = InternalProcessMonitor()

    def _get_tier(self, complexity: int) -> Optional[str]:
        """Determine which tier a piece of information belongs to based on its complexity."""
        for tier, (min_comp, max_comp) in self.TIERS.items():
            if min_comp <= complexity <= max_comp:
                return tier
        return None

    def create(self, key: str, value: Any, complexity: int) -> bool:
        """Add a new entry to the knowledge base."""
        tier = self._get_tier(complexity)
        if tier is None:
            return False
        with self.lock:
            if key not in self.knowledge_bases[tier]:
                self.knowledge_bases[tier][key] = value
                self.recent_updates.append((tier, key, value, complexity))
                self.knowledge_base_monitor.on_knowledge_update(tier, key, value, complexity)
                return True
            return False

    def read(self, key: str, complexity_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """Retrieve an entry from the knowledge base."""
        results = {}
        with self.lock:
            if complexity_range:
                min_comp, max_comp = complexity_range
                relevant_tiers = [tier for tier in self.TIERS.keys() if not (self.TIERS[tier][1] < min_comp or self.TIERS[tier][0] > max_comp)]
            else:
                relevant_tiers = self.TIERS.keys()
                
            for tier in relevant_tiers:
                if key in self.knowledge_bases[tier]:
                    results[tier] = self.knowledge_bases[tier][key]
            return results

    def update(self, key: str, value: Any, complexity: int) -> bool:
        """Update an existing entry in the knowledge base."""
        tier = self._get_tier(complexity)
        if tier is None:
            return False
        with self.lock:
            if key in self.knowledge_bases[tier]:
                self.knowledge_bases[tier][key] = value
                self.recent_updates.append((tier, key, value, complexity))
                self.knowledge_base_monitor.on_knowledge_update(tier, key, value, complexity)
                return True
            return False

    def delete(self, key: str) -> bool:
        """Delete an entry from the knowledge base."""
        with self.lock:
            for tier in self.TIERS.keys():
                if key in self.knowledge_bases[tier]:
                    del self.knowledge_bases[tier][key]
                    return True
            return False

    def get_recent_updates(self, n: int = None) -> List[Tuple[str, str, Any]]:
        """Get recent updates across all tiers."""
        with self.lock:
            updates = list(self.recent_updates)
            if n is not None:
                updates = updates[-n:]
            return updates

    def get_tier_stats(self) -> Dict[str, int]:
        """Get statistics about how many items are stored in each tier."""
        with self.lock:
            return {tier: len(kb) for tier, kb in self.knowledge_bases.items()}

# End of knowledge_base.py