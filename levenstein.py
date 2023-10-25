class levenstein_distance():
    def __init__(self, origin, goal, insertion_cost=1, deletion_cost=1, substitution_cost=2):
        self.origin=origin
        self.goal=goal
        self.insertion_cost=insertion_cost
        self.deletion_cost=deletion_cost
        self.substitution_cost=substitution_cost
        self.distance, self.insertions, self.deletions, self.substitutions, self.operations_performed=self.minimum_edit()

    def __str__(self):
        print("Minimum edit distance : {}".format(distance))
        print("Number of insertions : {}".format(insertions))
        print("Number of deletions : {}".format(deletions))
        print("Number of substitutions : {}".format(substitutions))
        print("Total number of operations : {}".format(insertions + deletions + substitutions))
        print("Actual Operations :")
        for i in range(len(operations_performed)):
            if operations_performed[i][0] == 'INS':
                print("{}) {} : {}".format(i + 1, operations_performed[i][0], operations_performed[i][1]))
            elif operations_performed[i][0] == 'DEL':
                print("{}) {} : {}".format(i + 1, operations_performed[i][0], operations_performed[i][1]))
            else:
                print("{}) {} : {} by {}".format(i + 1, operations_performed[i][0], operations_performed[i][1], operations_performed[i][2]))

    def minimum_edit(self):
        dp = [[0] * (len(self.origin) + 1) for i in range(len(self.goal) + 1)]
        for i in range(1, len(self.goal) + 1):
            dp[i][0] = dp[i - 1][0] + self.insertion_cost
        for i in range(1, len(self.origin) + 1):
            dp[0][i] = dp[0][i - 1] + self.deletion_cost

        operations = []

        for i in range(1, len(self.goal) + 1):
            for j in range(1, len(self.origin) + 1):
                if self.origin[j - 1] == self.goal[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + self.insertion_cost, dp[i - 1][j - 1] + self.substitution_cost, dp[i][j - 1] + self.deletion_cost)
        i = len(self.goal)
        j = len(self.origin)

        while (i != 0 and j != 0):
            if self.goal[i - 1] == self.origin[j - 1]:
                i = i - 1
                j = j - 1
            else:
                if dp[i][j] == dp[i - 1][j - 1] + self.substitution_cost:
                    operations.append(('SUB', self.origin[j - 1], self.goal[i - 1]))
                    i = i - 1
                    j = j - 1
                elif dp[i][j] == dp[i - 1][j] + self.insertion_cost:
                    operations.append(('INS', self.origin[i - 1]))
                    i = i - 1
                else:
                    operations.append(('DEL', self.origin[j - 1]))
                    j = j - 1
        while (j != 0):
            operations.append(('DEL', self.origin[j - 1]))
            j = j - 1
        while (i != 0):
            operations.append(('INS', self.origin[i - 1]))
            i = i - 1
        operations.reverse()
        distance=dp[len(self.goal)][len(self.origin)]

        insertions, deletions, substitutions = 0, 0, 0
        
        for i in operations:
            if i[0] == 'INS':
                insertions += 1
            elif i[0] == 'DEL':
                deletions += 1
            else:
                substitutions += 1
        return distance, insertions, deletions, substitutions, operations
