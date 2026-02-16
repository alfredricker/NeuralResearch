from src.group.base import Group

class ModelGroup(Group):
    def __init__(self, n: int, theta: float, id: int = 0):
        group_id = f'm{id}'
        super().__init__(n, theta, group_id) # creates neurons for the group