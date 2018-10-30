block(b1).
block(b2).
block(b3).
block(b4).
block(b5).
block(b6).
block(b7).
block(b8).

table(t1).
table(t2).
table(t3).
table(t4).
table(t5).
table(t6).
table(t7).
table(t8).

% 每个test的开始状态、目标状态和涉及到的对象
start_state1([clear(b2), on(b2, b1), on(b1, b3), on(b3, t1), clear(t2), clear(t3)]).
end_state1([clear(b3), on(b3, b1), on(b1, t1), clear(b2), on(b2, t2), clear(t3)]).
object_state1([b1, b2, b3, t1, t2, t3]).

start_state2([clear(b1), on(b1, b5), on(b5, b2), 
            on(b2, t1), clear(b3), on(b3, b4),
            on(b4, t2), clear(t3), clear(t4), clear(t5)]).
end_state2([clear(t1), clear(b2), on(b2, b1), 
            on(b1, b3), on(b3, t2), clear(t3), 
            clear(b4), on(b4, b5), on(b5, t4), clear(t5)]).
object_state2([b1, b2, b3, b4, b5, t1, t2, t3, t4, t5]).

start_state3([clear(b1), on(b1, b5), on(b5, b2), 
            on(b2, t1), clear(b3), on(b3, b4),
            on(b4, t2), clear(t3), clear(t4), clear(t5)]).
end_state3([clear(t1), clear(b4), on(b4, b3), 
            on(b3, b5), on(b5, b1), on(b1, b2), 
            on(b2, t2), clear(t3), clear(t4), clear(t5)]).
object_state3([b1, b2, b3, b4, b5, t1, t2, t3, t4, t5]).

start_state4([clear(b1), on(b1, t1), clear(t2),
            clear(b6), on(b6, b2), on(b2, b3),
            on(b3, t3), clear(t4), clear(b4),
            on(b4, b5), on(b5, t5), clear(t6)]).
end_state4([clear(b6), on(b6, b2), on(b2, b4),
            on(b4, b1), on(b1, b3), on(b3, b5),
            on(b5, t1), clear(t2), clear(t3),
            clear(t4), clear(t5), clear(t6)]).
object_state4([b1, b2, b3, b4, b5, b6, t1, t2, t3, t4, t5, t6]).

start_state5([clear(b1), on(b1, t1), clear(t2), 
            clear(b6), on(b6, b2), on(b2, b3),
            on(b3, t3), clear(t4), clear(b4),
            on(b4, b5), on(b5, t5), clear(b8),
            on(b8, b7), on(b7, t6), clear(t7),
            clear(t8)]).
end_state5([clear(b7), on(b7, b2), on(b2, b4),
            on(b4, b1), on(b1, b3), on(b3, b6),
            on(b6, b8), on(b8, b5), on(b5, t1),
            clear(t2), clear(t3), clear(t4),
            clear(t5), clear(t6), clear(t7),
            clear(t8)]).
object_state5([b1, b2, b3, b4, b5, b6, b7, b8, t1, t2, t3, t4, t5, t6, t7, t8]).

object(X):-block(X); table(X).

can(move(NowBlock, From, To), Objects, [clear(NowBlock), clear(To), on(NowBlock, From)]):-
    block(NowBlock),
    object(To),
    object(From),
    member(NowBlock, Objects),
    member(From, Objects),
    member(To, Objects),
    To\==NowBlock,
    From\==NowBlock,
    From\==To.
adds(move(NowBlock, From, To), [on(NowBlock, To), clear(From)]).
deletes(move(NowBlock, From, To), [on(NowBlock, From), clear(To)]).

member(X, [X | _]).
member(X, [_ | Tail]):-member(X, Tail).

conc([], L, L).
conc([X | L1], L2, [X | L3]):-conc(L1, L2, L3).

satisfied(_, []).
satisfied(State, [Goal | Goals]):-
    member(Goal, State), 
    satisfied(State, Goals).
select(State, Goals, Goal):-
    member(Goal, Goals), 
    not(member(Goal, State)).

achieves(Action, Goal):-
    adds(Action, Goals), 
    member(Goal, Goals).

delete_all([], _, []).
delete_all([X | L1], L2, Diff):-
    member(X, L2), 
    !,
    delete_all(L1, L2, Diff).
delete_all([X | L1], L2, [X | Diff]):-
    delete_all(L1, L2, Diff).

apply(State, Action, NewState):-
    deletes(Action, DelList),
    delete_all(State, DelList, State1),
    !,
    adds(Action, AddList),
    conc(AddList, State1, NewState).

% 判断一个序列是不是另一个序列的子序列
sublist([], _).
sublist([X | Tail], State):-
    member(X, State),
    sublist(Tail, State).
% 判断两个序列的元素是否都相同
is_equal_list(X, Y):-
    length(X, L1),
    length(Y, L2),
    L1 == L2,
    sublist(X, Y).

% 拼接序列
adds_list(L1, L2, [L1 | L2]).
% 删除序列头部
del_list([_ | Tail], Tail).
get_list_front([X | _], X).

% 判断一个序列是否为一个序列的序列的成员
not_list_member(_, []).
not_list_member(X, [Y | Tail]):-
    length(X, L1),
    length(Y, L2),
    L1 == L2,
    not(sublist(X, Y)),
    not_list_member(X, Tail).

% 判断两个状态（存在队列中）是否一致
is_equal_queue_state([S1, A1, D1, H1], [S2, A2, D2, H2]):-
    is_equal_list(S1, S2),
    is_equal_list(A1, A2),
    D1 == D2,
    H1 == H2.

% 判断一个状态是否在队列中
not_queue_state_member(_, []).
not_queue_state_member(X, [Y | Tail]):-
    length(X, L1),
    length(Y, L2),
    L1 == L2,
    not(is_equal_queue_state(X, Y)),
    not_queue_state_member(X, Tail).

% 得到可行的动作
achieves_vaild_moves(State, Objects, VaildAction):-
    can(VaildAction, Objects, Condition),
    sublist(Condition, State).

% 添加一个元素到队列末尾
adds_back(X, [], [X]).
adds_back(X, [Y | Tail], [Y | NextTail]):-
    adds_back(X, Tail, NextTail).

% 判断一个动作是否可行
is_valid_moves(MidState, NextAction, NextH, State, Goals, PreAction, Objects, NextVisitedState, Depth):-
    achieves_vaild_moves(State, Objects, Action), 
    apply(State, Action, MidState),  
    not_list_member(MidState, NextVisitedState),
    adds_back(Action, PreAction, NextAction),
    calh(Objects, NextH, MidState, Goals, Depth).

% 拷贝队列
copy([], []).
copy([X | Tail], [X | Tail2]):-
    copy(Tail, Tail2).

% 以PivotH为中心分割队列（分割成大于PivotH和小于等于PivotH的）
divide_list(_, [], [], []).
divide_list(PivotH, [[_, _, _, NowH] | Other], [[_, _, _, NowH] | Rest1], L2):-
    NowH =< PivotH,
    !, divide_list(PivotH, Other, Rest1, L2).
divide_list(PivotH, [[_, _, _, NowH] | Other], L1, [[_, _, _, NowH] | Rest2]):-
    NowH > PivotH,
    !, divide_list(PivotH, Other, L1, Rest2).

% 基于队列的A*
my_plan_a_star(Queue, Goals, _, Plan, _):-
    min_queue_state(Queue, [[State, PreAction, _, _] | _]),
    satisfied(State, Goals),
    copy(PreAction, Plan),
    !.
my_plan_a_star(Queue, Goals, Objects, Plan, VisitedState):-
    min_queue_state(Queue, [[State, PreAction, Depth, _] | Rest]),
    adds_list(State, VisitedState, NextVisitedState),
    NextDepth is Depth + 1,
    findall([MidState, NextAction, NextDepth, NextH], 
            is_valid_moves(MidState, NextAction, NextH, State, Goals, PreAction, Objects, NextVisitedState, NextDepth), 
            ValidQueueState),
    append(Rest, ValidQueueState, NextQueue),
    my_plan_a_star(NextQueue, Goals, Objects, Plan, NextVisitedState).

% 求解函数
my_plan(State, Goals, Objects, Plan):-
    calh(Objects, HVal, State, Goals, 0),
    conc([[State, [], 0, HVal]], [], Queue),
    my_plan_a_star(Queue, Goals, Objects, Plan, []).

% 找到队列里H最小的状态
min_queue_state([H|T], Result):-
    hdMin(H, [], T, Result).
hdMin(H, S, [], [H|S]).
hdMin(C, S, [H|T], Result):- 
    lessthan(C, H), 
    !, hdMin(C, [H|S], T, Result);
    hdMin(H, [C|S], T, Result).
lessthan([_, _, _, H1], [_, _, _, H2]):- 
    H1 =< H2.

% 判断Y是否在X下方
below(X, X, _).
below(Y, X, State):-
    Y \== X,
    block(Y),
    member(on(Z, Y), State),
    below(Z, X, State).

% 判断X是否在目标状态
isGoalPosition(X, State, Goals):-
    table(X);
    (member(on(X, Y), State),
    member(on(X, Y), Goals),
    isGoalPosition(Y, State, Goals)).

h(Val):- Val is 0.

% T01的h1
h1(Objects, Val, State, Goals):-
    findall(X, (member(X, Objects), block(X), not(isGoalPosition(X, State, Goals))), NotGoalList),
    length(NotGoalList, Val),
    !.

% T01的h2
h2(Objects, Val, State, Goals):-
    findall(X, (member(X, Objects), block(X), not(isGoalPosition(X, State, Goals)), below(Y, X, State), below(Y, X, Goals)), NotGoalList),
    length(NotGoalList, Val),
    !.

% 计算H值
calh(Objects, Val, State, Goals, Depth):-
    h1(Objects, V1, State, Goals),
    h2(Objects, V2, State, Goals),
    Val is V1 + V2 + Depth.