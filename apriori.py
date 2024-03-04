class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = None

    def _calculate_support(self, transactions, itemset):
        count = 0
        for transaction in transactions:
            if itemset.issubset(transaction):
                count += 1
        return count / len(transactions)

    def _generate_candidates(self, itemset, k):
        candidates = set()
        for item1 in itemset:
            for item2 in itemset:
                union = item1.union(item2)
                if len(union) == k and union not in candidates:
                    candidates.add(union)
        return candidates

    def _prune_candidates(self, transactions, candidates):
        pruned_candidates = set()
        for candidate in candidates:
            support = self._calculate_support(transactions, candidate)
            if support >= self.min_support:
                pruned_candidates.add(candidate)
        return pruned_candidates

    def _generate_itemsets(self, transactions):
        itemsets = [
            frozenset([item]) for transaction in transactions for item in transaction
        ]
        k = 2
        while True:
            candidates = self._generate_candidates(set(itemsets), k)
            candidates = self._prune_candidates(transactions, candidates)
            if not candidates:
                break
            itemsets.extend(candidates)
            k += 1
        return itemsets

    def _generate_rules(self, transactions):
        rules = []
        for itemset in self.itemsets:
            if len(itemset) < 2:
                continue
            for item in itemset:
                antecedent = itemset - set([item])
                consequent = set([item])
                confidence = self._calculate_support(
                    transactions, itemset
                ) / self._calculate_support(transactions, antecedent)
                if confidence >= self.min_confidence:
                    rules.append((antecedent, consequent, confidence))
        return rules

    def fit(self, transactions):
        self.itemsets = self._generate_itemsets(transactions)

    def get_itemsets(self):
        return self.itemsets

    def get_rules(self, transactions):
        return self._generate_rules(transactions)


# Example usage
transactions = [
    {"bread", "milk"},
    {"bread", "diaper", "beer", "egg"},
    {"milk", "diaper", "beer", "cola"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "cola"},
]

apriori = Apriori(min_support=0.5, min_confidence=0.5)
apriori.fit(transactions)

print("Frequent Itemsets:")
for itemset in apriori.get_itemsets():
    print(itemset)

# print("\nAssociation Rules:")
# for antecedent, consequent, confidence in apriori.get_rules(transactions):
#     print(f"{antecedent} => {consequent} (confidence: {confidence:.2f})")
