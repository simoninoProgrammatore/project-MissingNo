from missingno_battle import move_score


def test_super_effective_beats_neutral():
    assert move_score(90, 2.0, stab=False) > move_score(90, 1.0, stab=False)


def test_stab_bonus():
    assert move_score(80, 1.0, stab=True) > move_score(80, 1.0, stab=False)


def test_immune_scores_zero():
    assert move_score(100, 0.0, stab=True) == 0


def test_accuracy_lowers_score():
    assert move_score(120, 1.0, stab=False, accuracy=0.7) < move_score(100, 1.0, stab=False)
