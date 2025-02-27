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

####### 2 eme partie du cours 2 : #######
# Attention difference entre Obligation Revisable et Obligation Variable
# Le coupon d'une revisable c'est entre EURIBOR/LIBOR
# Le coupon d'un variable est connu a la fin et c'et sur un OIS souvent.

# Coupon =  (alpha*I + m) * D_i
# Alpha c'est la marge multiplicative
# I c'est la cotation de l'indice
# m c'est la marge additionelle
# D_i c'est le taux

# 2) Taux forward
# Soit on utilise les taux composé = F(t1,t2) = ((1+r2)^t_2)/(1+r1)^t_1)^(1/(t2-t1)) - 1

#3) Taux continu
# taux = (r2*t2 - r1*t1)/(t2-t1)


#4) Swap IRS



# Construciton du code python
# Le projet serar dans l'idee d'un tableau de bord de pricing
# Class Utils -> Maturity
# -> RATE (courbe de taux) -> il faudra sortir le taux sport et les taux fwd
#
# Clas Produit :
# Zc
# Portfolio : Fixed Income, Float Income, Swap, Currency swap
#

# La semaine pro on va lancer Streamit qui va permette de faire des tableau de bord












