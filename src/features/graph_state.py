"""
Feature Pipeline - Graph State.

Maintains an incrementally-updated graph: account/device/IP as nodes, edges
added as transactions occur. Degree/centrality features are READ from current
graph state, not recomputed fully each transaction - this is what makes ring
signals (device/IP-sharing) cheap to query at scoring time.

Uses networkx for simplicity. For a hackathon-scale synthetic dataset
(thousands, not millions, of nodes) this is fast enough to run inline.
"""

import networkx as nx


class GraphState:
    def __init__(self):
        self.graph = nx.Graph()

    def _node(self, kind: str, value) -> str:
        """Namespaced node id so 'device:abc' and 'ip:abc' never collide."""
        return f"{kind}:{value}"

    def add_transaction(self, account_id: str, device_id, ip_address) -> None:
        """
        Adds edges: account<->device, account<->ip (when present - respects
        MNAR nulls, a nulled device_fingerprint simply adds no edge that round,
        which is itself meaningful: a low-degree/isolated account with no
        device history is a distinct graph state from a well-connected one).
        """
        acct_node = self._node("account", account_id)
        self.graph.add_node(acct_node, kind="account")

        if device_id:
            dev_node = self._node("device", device_id)
            self.graph.add_node(dev_node, kind="device")
            self.graph.add_edge(acct_node, dev_node)

        if ip_address:
            ip_node = self._node("ip", ip_address)
            self.graph.add_node(ip_node, kind="ip")
            self.graph.add_edge(acct_node, ip_node)

    def account_features(self, account_id: str) -> dict:
        """
        Read current graph state for an account. Returns structural zeros
        (not nulls) for brand-new/unseen accounts - a real, meaningful state
        per the spec's missing-data handling rules.
        """
        acct_node = self._node("account", account_id)

        if acct_node not in self.graph:
            return {
                "graph_degree": 0,
                "graph_shared_device_count": 0,
                "graph_shared_ip_count": 0,
                "graph_2hop_account_count": 0,
            }

        degree = self.graph.degree[acct_node]

        # count OTHER accounts reachable within 2 hops (account -> device/ip -> other account)
        # this is the ring signal: many distinct accounts sharing one device/IP
        neighbors_1hop = set(self.graph.neighbors(acct_node))
        accounts_2hop = set()
        shared_device_count = 0
        shared_ip_count = 0

        for n in neighbors_1hop:
            node_kind = self.graph.nodes[n].get("kind")
            n_neighbors = list(self.graph.neighbors(n))
            other_accounts = [x for x in n_neighbors if x != acct_node and self.graph.nodes[x].get("kind") == "account"]
            accounts_2hop.update(other_accounts)
            if node_kind == "device" and other_accounts:
                shared_device_count += len(other_accounts)
            if node_kind == "ip" and other_accounts:
                shared_ip_count += len(other_accounts)

        return {
            "graph_degree": degree,
            "graph_shared_device_count": shared_device_count,
            "graph_shared_ip_count": shared_ip_count,
            "graph_2hop_account_count": len(accounts_2hop),
        }

    def get_2hop_subgraph(self, account_id: str) -> nx.Graph:
        """Returns the 2-hop ego subgraph around an account - what the GNN component consumes."""
        acct_node = self._node("account", account_id)
        if acct_node not in self.graph:
            return nx.Graph()
        return nx.ego_graph(self.graph, acct_node, radius=2)


if __name__ == "__main__":
    gs = GraphState()

    # simulate a mule ring: 4 different accounts sharing one device
    for i in range(4):
        gs.add_transaction(f"mule_{i}", device_id="ring_device_xyz", ip_address=f"ip_{i}")

    # one normal, isolated account
    gs.add_transaction("normal_user", device_id="own_device", ip_address="own_ip")

    print("mule_0 features:", gs.account_features("mule_0"))
    print("normal_user features:", gs.account_features("normal_user"))
    print("unseen account features:", gs.account_features("never_seen"))
