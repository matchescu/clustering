import numpy as np

class LearnedStringDistance:
    def __init__(self, alphabet, smoothing=1e-12):
        self.alphabet = alphabet
        self.smoothing = smoothing
        self.init_params()

    def init_params(self):
        # Initialize transition probabilities
        self.trans_probs = {
            'M': {'M': 0.9, 'I': 0.05, 'D': 0.05},
            'I': {'I': 0.9, 'M': 0.1},
            'D': {'D': 0.9, 'M': 0.1}
        }

        # Initialize emission probabilities
        self.emit_probs = {
            'M': {a: {b: self.smoothing for b in self.alphabet} for a in self.alphabet},
            'I': {a: self.smoothing for a in self.alphabet},
            'D': {a: self.smoothing for a in self.alphabet}
        }

        # Set initial probabilities for matches
        for a in self.alphabet:
            self.emit_probs['M'][a][a] = 1 - (len(self.alphabet) - 1) * self.smoothing

    def forward(self, x, y):
        T, V = len(x), len(y)
        M = np.zeros((T + 1, V + 1))
        I = np.zeros((T + 1, V + 1))
        D = np.zeros((T + 1, V + 1))

        M[0, 0] = 1

        for i in range(1, T + 1):
            D[i, 0] = D[i - 1, 0] * self.trans_probs['D']['D'] * self.emit_probs['D'][x[i - 1]]

        for j in range(1, V + 1):
            I[0, j] = I[0, j - 1] * self.trans_probs['I']['I'] * self.emit_probs['I'][y[j - 1]]

        for i in range(1, T + 1):
            for j in range(1, V + 1):
                M[i, j] = (M[i - 1, j - 1] * self.trans_probs['M']['M'] +
                           I[i - 1, j - 1] * self.trans_probs['I']['M'] +
                           D[i - 1, j - 1] * self.trans_probs['D']['M']) * self.emit_probs['M'][x[i - 1]][y[j - 1]]

                I[i, j] = (M[i, j - 1] * self.trans_probs['M']['I'] +
                           I[i, j - 1] * self.trans_probs['I']['I']) * self.emit_probs['I'][y[j - 1]]

                D[i, j] = (M[i - 1, j] * self.trans_probs['M']['D'] +
                           D[i - 1, j] * self.trans_probs['D']['D']) * self.emit_probs['D'][x[i - 1]]

        return M[T, V] + I[T, V] + D[T, V]

    def backward(self, x, y):
        T, V = len(x), len(y)
        M = np.zeros((T + 1, V + 1))
        I = np.zeros((T + 1, V + 1))
        D = np.zeros((T + 1, V + 1))

        M[T, V] = I[T, V] = D[T, V] = 1

        for i in range(T - 1, -1, -1):
            D[i, V] = D[i + 1, V] * self.trans_probs['D']['D'] * self.emit_probs['D'][x[i]]

        for j in range(V - 1, -1, -1):
            I[T, j] = I[T, j + 1] * self.trans_probs['I']['I'] * self.emit_probs['I'][y[j]]

        for i in range(T - 1, -1, -1):
            for j in range(V - 1, -1, -1):
                M[i, j] = (M[i + 1, j + 1] * self.trans_probs['M']['M'] +
                           I[i + 1, j + 1] * self.trans_probs['I']['M'] +
                           D[i + 1, j + 1] * self.trans_probs['D']['M']) * self.emit_probs['M'][x[i]][y[j]]

                I[i, j] = (M[i, j + 1] * self.trans_probs['M']['I'] +
                           I[i, j + 1] * self.trans_probs['I']['I']) * self.emit_probs['I'][y[j]]

                D[i, j] = (M[i + 1, j] * self.trans_probs['M']['D'] +
                           D[i + 1, j] * self.trans_probs['D']['D']) * self.emit_probs['D'][x[i]]

        return M[0, 0]

    def expectation_step(self, x_pairs):
        expectations = {
            'transitions': {state: {next_state: self.smoothing for next_state in next_states}
                            for state, next_states in self.trans_probs.items()},
            'emissions': {state: {a: {b: self.smoothing for b in (self.alphabet if state == 'M' else [None])}
                                  for a in (self.alphabet if state != 'None' else [None])}
                          for state in ['M', 'I', 'D']}
        }

        total_prob = sum(self.forward(x[0], x[1]) for x in x_pairs)

        for x_pair in x_pairs:
            x, y = x_pair
            x_tilde_prob = total_prob / len(x_pairs)
            fwd_prob = np.log(self.forward(x, y))
            bwd_prob = np.log(self.backward(x, y))

            T_x_tilde_prob = fwd_prob / x_tilde_prob
            B_x_tilde_prob = bwd_prob / x_tilde_prob

            expectations['transitions']['M']['M'] += T_x_tilde_prob
            expectations['transitions']['M']['I'] += B_x_tilde_prob
            expectations['transitions']['M']['D'] += B_x_tilde_prob

            for char_x in x:
                for char_y in y:
                    expectations['emissions']['M'][char_x][char_y] += T_x_tilde_prob
                    #expectations['emissions']['I'][None][char_y] += B_x_tilde_prob
                    #expectations['emissions']['D'][char_x][None] += B_x_tilde_prob

        return expectations

    def maximization_step(self, expectations):
        total_transitions = sum(sum(next_states.values()) for next_states in expectations['transitions'].values())

        for state in expectations['transitions']:
            total_state_transitions = sum(expectations['transitions'][state].values())

            for next_state in expectations['transitions'][state]:
                self.trans_probs[state][next_state] = expectations['transitions'][state][next_state] / total_state_transitions

        for state in expectations['emissions']:
            for a in expectations['emissions'][state]:
                total_emissions = sum(expectations['emissions'][state][a].values())

                for b in expectations['emissions'][state][a]:
                    self.emit_probs[state][a][b] = expectations['emissions'][state][a][b] / total_emissions

    def fit(self, x_pairs, max_iter=100):
        for _ in range(max_iter):
            expectations = self.expectation_step(x_pairs)
            self.maximization_step(expectations)

    def compute_distance(self, x, y):
        return -np.log(self.forward(x, y))


def test_basic():
    # Example usage
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    learned_distance = LearnedStringDistance(alphabet)
    x_pairs = [('hello', 'hallo'), ('world', 'word')]
    learned_distance.fit(x_pairs)
    distance = learned_distance.compute_distance('andrei', 'andi')

    assert distance > 0
