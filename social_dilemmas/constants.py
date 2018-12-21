# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means apply spawn point
# '' is empty space

HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

CLEANUP_MAP = [
'@@@@@@@@@@@@@@@@@@'
'@FFFFFF     BBBBB@'
'@HHHHHH      BBBB@'
'@FFFFFF     BBBBB@'
'@FFFFF  P    BBBB@'
'@FFFFF    P BBBBB@'
'@HHHHH       BBBB@'
'@FFFFF      BBBBB@'
'@HHHHHHSSSSSSBBBB@'
'@HHHHHHSSSSSSBBBB@'
'@FFFFF   P P BBBB@'
'@HHHHH   P  BBBBB@'
'@FFFFFF    P BBBB@'
'@HHHHHH P   BBBBB@'
'@FFFFF       BBBB@'
'@HHHH    P  BBBBB@'
'@FFFFF       BBBB@'
'@HHHHH  P P BBBBB@'
'@FFFFF       BBBB@'
'@HHHH       BBBBB@'
'@FFFFF       BBBB@'
'@HHHHH      BBBBB@'
'@FFFFF       BBBB@'
'@HHHH       BBBBB@'
'@@@@@@@@@@@@@@@@@@']