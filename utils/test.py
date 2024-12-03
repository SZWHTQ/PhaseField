class Parent1:
    def __init__(self):
        self.name = "Parent1"

    def method(self):
        print("Method from Parent1")

    def __call__(self):
        self.method()


class Parent2:
    def __init__(self):
        self.name = "Parent2"

    def method(self):
        print("Method from Parent2")

    def __call__(self):
        self.method()


class Child(Parent1, Parent2):
    def __init__(self):
        Parent1.__init__(self)  # Initialize Parent1's __init__
        Parent2.__init__(self)  # Initialize Parent2's __init__
        self.name = "Child"  # Override the name in the child class

    def method(self):
        super().method()  # Calls the next method in the MRO
        print("Method from Child")

    def get_parent_names(self):
        # Access names from both Parent1 and Parent2
        parent1_name = Parent1.__getattribute__(self, "name")
        parent2_name = Parent2.__getattribute__(self, "name")
        return parent1_name, parent2_name


print(Child.mro())  # Method Resolution Order (MRO)

child = Child()

# Access names from Parent1 and Parent2
parent1_name, parent2_name = child.get_parent_names()
print(f"Parent1 name: {parent1_name}")
print(f"Parent2 name: {parent2_name}")

# Output the overridden name in Child
print(f"Child name: {child.name}")

child()
