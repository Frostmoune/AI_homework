male(george).
female(mum).
husband(george, mum).
wife(mum, george).
child(elizabeth, george).
child(elizabeth, mum).
child(margaret, george).
child(margaret, mum).

female(margaret).

male(spencer).
female(kydd).
husband(spencer, kydd).
wife(kydd, spencer).
child(diana, spencer).
child(diana, kydd).

female(elizabeth).
male(philip).
husband(philip, elizabeth).
wife(elizabeth, philip).
child(charles, elizabeth).
child(charles, philip).
child(anne, elizabeth).
child(anne, philip).
child(andrew, elizabeth).
child(andrew, philip).
child(edward, elizabeth).
child(edward, philip).

female(diana).
male(charles).
husband(charles, diana).
wife(diana, charles).
child(william, diana).
child(william, charles).
child(harry, diana).
child(harry, charles).

female(anne).
male(mark).
husband(mark, anne).
wife(anne, mark).
child(peter, anne).
child(peter, mark).
child(zara, anne).
child(zara, mark).

female(sarah).
male(andrew).
husband(andrew, sarah).
wife(sarah, andrew).
child(beatrice, sarah).
child(beatrice, andrew).
child(eugenie, sarah).
child(eugenie, andrew).

female(sophie).
male(edward).
husband(edward, sophie).
wife(sophie, edward).
child(louise, sophie).
child(louise, edward).
child(james, sophie).
child(james, edward).

male(william).
male(harry).

male(peter).
female(zara).

female(beatrice).
female(eugenie).

female(louise).
male(james).

daughter(X, Y):-child(X, Y), female(X).
son(X, Y):-child(X, Y), male(X).
father(X, Y):-child(Y, X), male(X).
mother(X, Y):-child(Y, X), female(X).
brother(X, Y):-male(X), father(Z, Y), father(Z, X), not(X == Y).
sister(X, Y):-female(X), father(Z, X), father(Z, Y), not(X == Y).

grandChild(X, Y):-child(Z, Y), child(X, Z).
greatGrandParent(X, Y):-child(Z, X), grandChild(Y, Z).
ancestor(X, Y):-child(Y, X); grandChild(Y, X); greatGrandParent(X, Y).

parentInLaw(X, Y):-(husband(Z, Y); wife(Z, Y)), child(Z, X).
brotherInLaw(X, Y):-((husband(Z, Y); wife(Z, Y)), brother(X, Z));
                        (sister(Z, Y), husband(X, Z));
                        ((husband(Z, Y); wife(Z, Y)), sister(W, Z), husband(X, W)).
sisterInLaw(X, Y):-((husband(Z, Y); wife(Z, Y)), sister(X, Z));
                    (brother(Z, Y), wife(X, Z));
                    ((husband(Z, Y); wife(Z, Y)), brother(W, Z), wife(X, W)).
firstCousin(X, Y):-grandChild(X, Z), grandChild(Y, Z), not(brother(X, Y)), not(sister(X, Y)), not(X == Y).

aunt(X, Y):-child(Y, Z), (sister(X, Z); sisterInLaw(X, Z)).
uncle(X, Y):-child(Y, Z), (brother(X, Z); brotherInLaw(X, Z)).

mySearch(X, Y, 1, 0, 1):-firstCousin(X, Y).
mySearch(X, Y, 1, N, L):-firstCousin(Z, Y), child(W, Z), N1 is N - 1, L1 is L - 1, mySearch(X, W, 1, N1, L1).
mySearch(X, Y, M, N, L):-((L == M), (N > 0), child(Y, Z), N1 is N - 1, mySearch(X, Z, M, N1, L));
                        ((L >= M), child(Y, Z), M1 is M - 1, N1 is N + 1, mySearch(X, Z, M1, N1, L)).
mthCousinNremoved(X, Y, M, N):-(M > 0), (mySearch(X, Y, M, N, M); mySearch(X, Y, M, N, M + N)).