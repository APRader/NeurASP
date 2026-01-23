% Suit values
suit_value(d,0). suit_value(c,13). suit_value(s,26). suit_value(h,39).

% Result calculation
result(X) :- suit(0,S0), suit(1,S1), suit(2,S2), suit(3,S3),
             suit_value(S0,SV0), suit_value(S1,SV1), suit_value(S2, SV2), suit_value(S3,SV3),
             rank(0,R0), rank(1,R1), rank(2,R2), rank(3,R3),
             rank_value(R0,V0), rank_value(R1,V1), rank_value(R2,V2), rank_value(R3,V3),
             X = V0 + SV0 + V1 + SV1 + V2 + SV2 + V3 + SV3.
:- result(X), result(Y), X!=Y.


% Neural predicate
nn(card(4,p), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]).