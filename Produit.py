from abc import ABC, abstractmethod

class Produit(ABC):
    def __init__(self, nom: str):
        self.nom = nom
    
    @abstractmethod
    def prix(self):
        """
        Méthode abstraite pour calculer le prix du produit.
        Chaque classe fille devra implémenter cette méthode.
        """
        pass
    
    def description(self):
        """
        Retourne une description du produit.
        """
        return f"Produit: {self.nom}"
    
    def __str__(self):
        return self.description()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(nom={self.nom})"
