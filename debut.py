# Premier cours de produit structué (1/6)
# C'est un projet PYTHON purement écrit 2 semaine apres la fin du dernier cours
# Sujet de Pricing ou on peut pricer tout quoi
# B(t+delta_t)  = B(t)*exp(delta_t*r)
# B(t+delta_t)  = B(t) * DF


# Pour une courbe de taux il vaut mieux travailler avec une interpo spline cubique qu'une interpo cuboc
# Sinon il existe le modele : Nelson Siegel
# r(t) = B0 + B1*(1-exp(lambda*delta_t)/(lambda*t)) + B2*((1-exp(-lambda*delta_t))/(lamnda*t)-exp(-lambda*t))

# Mais en realite il vaudrait mieux utliser SVENSON pour interpoler

# On peut installer GITHUB DESKTOP et il faut le mapper au compte
# Une fois installer on a tous les projets existant
# Il faut vraiment apprendre GITHUB askip les entreprises aime bien





##################### COURS NUMERO 2 #########################

# Crrer un environement:  python3 -m venv venv
# Activer l'environement: source venv/bin/activate
# pip install scipy
# pip list
# Puis faire : pip freeze > requirements.txt  Ca c'est bien ca permet de faire un fichier txt avec les bonnes libraires
# La personne qui veut l'utliser doit faire : pip install -r requirements.txt




