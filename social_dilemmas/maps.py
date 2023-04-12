# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means apple spawn point
# ' ' is empty space

HARVEST_MAP = [
    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
    "@ P   P      A    P AAAAA    P  A P  @",
    "@  P     A P AA    P    AAA    A  A  @",
    "@     A AAA  AAA    A    A AA AAAA   @",
    "@ A  AAA A    A  A AAA  A  A   A A   @",
    "@AAA  A A    A  AAA A  AAA        A P@",
    "@ A A  AAA  AAA  A A    A AA   AA AA @",
    "@  A A  AAA    A A  AAA    AAA  A    @",
    "@   AAA  A      AAA  A    AAAA       @",
    "@ P  A       A  A AAA    A  A      P @",
    "@A  AAA  A  A  AAA A    AAAA     P   @",
    "@    A A   AAA  A A      A AA   A  P @",
    "@     AAA   A A  AAA      AA   AAA P @",
    "@ A    A     AAA  A  P          A    @",
    "@       P     A         P  P P     P @",
    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
]

CLEANUP_MAP = [
    "@@@@@@@@@@@@@@@@@@",
    "@HHHHHH     BBBBB@",
    "@RRRRRR      BBBB@",
    "@HHHHHH     BBBBB@",
    "@RRRRR  P    BBBB@",
    "@HHHHH    P BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH      BBBBB@",
    "@SSSSSSHHHHHHBBBB@",
    "@SSSSSSHHHHHHBBBB@",
    "@HHHHH   P P BBBB@",
    "@RRRRR   P  BBBBB@",
    "@HHHHHH    P BBBB@",
    "@RRRRRR P   BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRR    P  BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRRR  P P BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRR       BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRRR      BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRR       BBBBB@",
    "@@@@@@@@@@@@@@@@@@",
]


# 'S' means turned-on switch
# 's' means turned-off switch
# 'D' means closed door
# 'd' means opened door
class SwitchMapElements:
    top_row = "@@@D@@@"
    empty_row = "@     @"
    one_switch_row = "@s    @"
    two_switch_row = "@s   s@"
    bottom_row = "@@@@@@@"
