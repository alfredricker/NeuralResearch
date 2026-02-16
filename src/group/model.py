from src.group.base import Group

class ModelGroup(Group):
    def __init__(self, n: int, theta: float, id: int = 0):
        super().__init__(n)
        self.theta = theta
        self.group_id = f'm{id}'