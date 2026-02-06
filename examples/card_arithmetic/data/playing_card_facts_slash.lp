player(p1). player(p2). player(p3).

% Suits
suit(h).
suit(s).
suit(d).
suit(c).

% Ranks
rank(a).
rank(2).
rank(3).
rank(4).
rank(5).
rank(6).
rank(7).
rank(8).
rank(9).
rank(10).
rank(j).
rank(q).
rank(k).

% Rank values
rank_value(2, 2).
rank_value(3, 3).
rank_value(4, 4).
rank_value(5, 5).
rank_value(6, 6).
rank_value(7, 7).
rank_value(8, 8).
rank_value(9, 9).
rank_value(10, 10).
rank_value(j, 11).
rank_value(q, 12).
rank_value(k, 13).
rank_value(a, 14).

% Converting from cards to suits and ranks
suit(P,h) :- card(0,+P,-C), C <= 12.
suit(P,c) :- card(0,+P,-C), C >= 13, C <= 25.
suit(P,s) :- card(0,+P,-C), C >= 26, C <= 38.
suit(P,d) :- card(0,+P,-C), C >= 39.
rank(P,R) :- card(0,+P,-C), rank_value(R,C\13+2).