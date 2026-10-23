from src.optimization.reporting import E3Record, format_e3_markdown, summarize_e3


def make_records():
    return [
        E3Record("room1", "random", 0.5, 0.6, False, True, 10, 0.2),
        E3Record("room2", "random", 0.7, 0.7, True, True, 10, 0.3),
        E3Record("room1", "greedy", 0.5, 0.8, False, True, 20, 0.4),
    ]


def test_e3_summary_and_markdown():
    records = make_records()
    summary = summarize_e3(records)

    assert summary["rooms"] == 2
    assert summary["algorithms"]["random"]["mean_delta"] == 0.05
    assert summary["algorithms"]["random"]["improved_rooms"] == 1
    assert summary["algorithms"]["greedy"]["feasible_best"] == 1

    markdown = format_e3_markdown(records, summary)
    assert "# E3: Layout Optimization" in markdown
    assert "| room1 | greedy |" in markdown
    assert "+0.3000" in markdown
