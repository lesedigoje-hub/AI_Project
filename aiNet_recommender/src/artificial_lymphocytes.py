class ARB:

    # Artificial Recognition Ball - a single antibody in the aiNet immune network 

    def __init__(self, vector: list[float], movie_id=None):
        self.vector = vector
        self.movie_id = movie_id  # the movie from which this arb came

        self.neighbours = []

        self.antigen_stimulation = 0.0
        self.network_stimulation = 0.0
        self.network_suppression = 0.0
        self.total_stimulation = 0.0

        self.resources = 0.0

    def __str__(self):
        vector = f'ARB vector: {self.vector}\n'
        stimulation = f'Total stimulation: {self.total_stimulation}\n'
        resources = f'Resources: {self.resources}\n'

        return vector + stimulation + resources
