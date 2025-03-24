from abc import ABC, abstractmethod
from datetime import datetime

class ProduitDerive(ABC):
    def __init__(self, sous_jacent: object, nom: str, sens: str, nominal: float):
        """
        Initialisation d'un produit dérivé.
        
        :param sous_jacent: objet : L'objet représentant le sous-jacent, par exemple une instance de la classe Action ou Taux
        :param nom: str : Le nom du produit dérivé
        :param sens: str : Le sens de la position, 'long' ou 'short'
        """
        if sens not in ["long", "short"]:
            raise ValueError("Le sens doit être 'long' ou 'short'.")
        
        self.sous_jacent = sous_jacent  # Le produit sous-jacent (par exemple une Action ou un Taux)
        self.nom = nom  # Le nom du produit dérivé
        self.sens = sens  # Sens de la position : 'long' ou 'short'
        self.nominal = nominal
    
    @abstractmethod
    def prix(self, date: datetime = None):
        """
        Méthode abstraite pour calculer le prix du produit dérivé.
        Chaque classe fille devra implémenter cette méthode.
        """
        pass
    
    def description(self):
        """
        Retourne une description du produit dérivé.
        """
        return f"Produit Dérivé: {self.nom}, Sens: {self.sens}, Sous-jacent: {self.sous_jacent.__class__.__name__}, Nominal: {self.nominal}"

    def __str__(self):
        return self.description()