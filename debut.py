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






##### Dernier cours

## 1 PRICING /Mesure risque :

#il Faut caclculer les greeks pour tout y comporis obligation
# Faire un payoff sous frome de graphique + essaye de determiner les zones de probas (proba d'exercice) c'est encore mieux
# Pour la mesure de risk , chose sympa a avoir c'est un stress scénario (faire passser la vol de 20% à 40%) et voir l'impact du changement de vol sur le produit (facile)


## 2 PARTIE FINANCE/PRODUITS :

# Il faut faire une douzaine de produit mais faire bien varier : elementaire (call/put actions et taux)
# faire strategie optionnelle (call spread,strip,strap,strandle)
# Et surtout faire un type de produit autocall (un peu plus complexe) --> MC par excellence
# L'idee cest de reutiliser ce qu'on a fait avant pour faire un nouveau produit
# Exemple un call spread doit etre facile a coder juste faire call_K1 - call_k2 en 1 ligne

## 3 PARTIE MODEELE:
# Faire du MC
# soit faire que de la vol sto, soit faire que un taux sto ou les deux
# Idee de faire un modele de diffusion avec une calibration sous jacente


#PARTIE IT :

# Absolument tout faire en ORIENTE OBJET pour les produit (pas oblige pour MC et partie graphique)
# Faire du FRONT END donc une cerrtaine visualisation (streamlite)
# On peut heberger sur le cloud (on peut mettre du streamlite) et montrer au recruteur


# Faire oblig -> maturite / convexite/ sensi / duration
# Faire swap  -> maturite / convexite/ sensi / duration

# Greeks














