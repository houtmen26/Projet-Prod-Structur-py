from Maturite import Maturite

# Création d'une instance de Maturite
maturite_obj = Maturite("2024-02-20", "2029-02-20", "Act/365")

# Récupération de la maturité
maturite = maturite_obj.maturite_en_annees  # Utilise l'attribut au lieu d'appeler la méthode directement
print(maturite)