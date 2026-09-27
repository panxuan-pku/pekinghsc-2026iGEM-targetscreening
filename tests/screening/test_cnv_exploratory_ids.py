"""Exploratory tables retain distinct IDs; the ranking entry stays symbol-unique."""
from pathlib import Path

import pandas as pd
import pytest

from screening.src.cnv import workflow as w
from screening.src.cnv.artifacts import StageRun
from .test_cnv_workflow import cfg, _out, _audit, _seed_infercnv


@pytest.mark.parametrize('include_auto', [False, True])
def test_auto_shared_symbol_preserves_ids_without_weakening_ranking_entry(cfg, include_auto):
    cfg['extract'].update(mode='cnv', include_auto_in_candidates=include_auto)
    order = Path(cfg['annotation']['gene_order_tsv'])
    genes = pd.read_csv(order, sep='\t')
    extra = pd.DataFrame({'gene_name': ['SHARED', 'SHARED'], 'gene_id': ['ID3', 'ID4'],
                          'chrom': ['chr1', 'chr1'], 'start': [500, 700], 'end': [600, 800],
                          'strand': ['+', '+']})
    pd.concat([genes, extra], ignore_index=True).to_csv(order, sep='\t', index=False)
    _seed_infercnv(cfg)
    seg = _out(cfg) / 'segments_all.csv'
    with StageRun(cfg, 'call_segments', [seg], w._stage_parameters(cfg, 'call_segments')) as run:
        run.upstream('infercnv', w._stage_parameters(cfg, 'infercnv'))
        pd.DataFrame([{'chr': 'chr1', 'start': 500, 'end': 800, 'resolution': 'standard'}]).to_csv(seg, index=False)
    if include_auto:
        with pytest.raises(ValueError, match='gene identity for SHARED'):
            w.cmd_extract_genes(cfg)
        assert _audit(cfg)['validation_status'] == 'ERROR'
        assert not (_out(cfg) / 'candidates.csv').exists()
        return
    for repeat in range(2):
        candidates, _, _ = w.cmd_extract_genes(cfg)
        assert candidates.gene_symbol.tolist() == ['A', 'B']
        assert candidates.columns.tolist() == w.CANDIDATE_COLUMNS
        for filename in ('auto_segment_genes.csv', 'candidates_all.csv'):
            table = pd.read_csv(_out(cfg) / filename)
            shared = table[table.gene_symbol == 'SHARED']
            assert shared.gene_id.tolist() == ['ID3', 'ID4']
            assert shared.start.tolist() == [500, 700]
        detail = pd.read_csv(_out(cfg) / 'candidate_provenance.csv')
        assert detail.loc[detail.source == 'auto_segment', 'in_candidates'].tolist() == [False, False]
        current = {p.name: p.read_bytes() for p in _out(cfg).glob('*.csv')}
        if repeat == 0:
            first = current
        else:
            assert current == first


@pytest.mark.parametrize('change', [{'gene_id': ''}, {'gene_id': 'ID1', 'start': 110}])
def test_exploratory_identity_must_still_be_complete_and_consistent(change):
    records = [{'gene_symbol': 'A', 'gene_id': 'ID1', 'chrom': 'chr1', 'start': 100,
                'end': 200, 'strand': '+', 'genome_build': 'GRCh38', 'source': 'auto_segment',
                'deletion_id': 'auto_1', 'interval_chrom': 'chr1', 'interval_start': 100, 'interval_end': 300}]
    records.append(dict(records[0], **change))
    with pytest.raises(ValueError, match='gene identity'):
        w._unique_candidates(pd.DataFrame(records), by_id=True)


def test_interval_validation_does_not_borrow_expression_from_same_symbol(cfg, monkeypatch):
    import anndata
    cfg['extract']['mode'] = 'cnv'
    cfg['known_intervals']['del1']['markers'] = ['A']
    order = Path(cfg['annotation']['gene_order_tsv'])
    genes = pd.read_csv(order, sep='\t')
    extra = dict(genes.iloc[0], gene_id='ID3', start=500, end=600)
    pd.concat([genes, pd.DataFrame([extra])], ignore_index=True).to_csv(order, sep='\t', index=False)
    signal = _out(cfg) / 'window_signal_standard.tsv'
    h5ad = _out(cfg) / 'cnv_input.h5ad'
    with StageRun(cfg, 'infercnv', [signal, h5ad], w._stage_parameters(cfg, 'infercnv')) as run:
        run.input('gene_order', order)
        pd.DataFrame({'chr': ['chr1'], 'start': [100], 'end': [600], 'score': [0.]}).to_csv(signal, sep='\t', index=False)
        h5ad.write_bytes(b'QC fixture')
    seg = _out(cfg) / 'segments_all.csv'
    with StageRun(cfg, 'call_segments', [seg], w._stage_parameters(cfg, 'call_segments')) as run:
        run.upstream('infercnv', w._stage_parameters(cfg, 'infercnv'))
        pd.DataFrame([{'chr': 'chr1', 'start': 500, 'end': 600, 'resolution': 'standard'}]).to_csv(seg, index=False)
    monkeypatch.setattr(anndata, 'read_h5ad', lambda path: object())
    qc = pd.DataFrame({'gene_name': ['A', 'B', 'A'], 'gene_id': ['ID1', 'ID2', 'ID3'],
                       'found': [False, True, True], 'log2fc_patient_vs_ref': [float('nan'), -0.5, 2.]})
    monkeypatch.setattr(w, 'interval_gene_qc', lambda *args, **kwargs: qc)
    qc['mean_counts_patient'] = [float('nan'), 1., 9.]
    qc['mean_counts_ref'] = [float('nan'), 1., 1.]
    qc['pct_expr_patient'] = [float('nan'), 0.5, 0.5]
    qc['pct_expr_ref'] = [float('nan'), 0.5, 0.5]
    qc['chrom'] = 'chr1'
    _, _, val = w.cmd_extract_genes(cfg)
    assert val['AC-CNV-2 (markers detected in data)'].tolist() == ['FAIL']
    assert val.n_genes_in_expression.tolist() == [1]
    assert val.mean_log2fc_interval_genes.tolist() == [-0.5]
    assert val.markers_in_expression.tolist() == ['']
    report = (_out(cfg) / 'cnv_report.md').read_text()
    assert '| A | chr1 | 9.000' not in report
    assert '| B | chr1 | 1.000' in report
