"""真实 Chromium 点击验收；需已运行的服务，不下载模型或浏览器。"""
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright-browsers"))
URL = f"http://127.0.0.1:{os.environ.get('VCT_PORT', '8377')}/"


def check_history(pg, expect):
    pg.locator("#historyBtn").click()
    frame = pg.frame_locator('iframe[title="历史实验与方法迭代（只读）"]')
    expect(frame.locator("#archiveState")).to_contain_text("只读材料已核对")
    expect(frame.locator("#wsMetric")).to_contain_text("-0.0456")
    expect(frame.locator("#wsMetric")).to_contain_text("0.0719")
    expect(frame.locator("#celloracle .pending")).to_contain_text("图片待补")
    expect(frame.locator("#gears .notice")).to_contain_text("不作为当前结论")
    for key in ("ms_figure", "gears_figure", "cipher_figure"):
        img = frame.locator(f'[data-image="{key}"] img')
        img.scroll_into_view_if_needed()
        expect(img).to_be_visible()
        expect(frame.locator(f'[data-image="{key}"] p')).to_contain_text("已保存原图")
        assert img.evaluate("el => el.complete && el.naturalWidth > 0")
    assert "?" not in pg.locator('iframe[title="历史实验与方法迭代（只读）"]').get_attribute("src")
    frame.locator("h1").scroll_into_view_if_needed()


def run_checks(pg, expect):
    pg.goto(URL, wait_until="domcontentloaded")
    panel = pg.locator("#encoderExploration")
    assert panel.get_attribute("open") is None, "实验功能应默认折叠"
    expect(pg.locator("#goBtn")).to_be_hidden()
    expect(pg.locator("#dsState")).to_contain_text("✓ PBMC 2700")
    pg.wait_for_function("typeof umapReady !== 'undefined' && umapReady")
    check_history(pg, expect)
    pg.locator(".tab").filter(has_text="UMAP").first.click()
    plot = pg.locator("#plot_umap")
    plot.scroll_into_view_if_needed()
    # Read the rendered trace/axes only, then send a real mouse click. Do not
    # call selectCell/pickGene or synthesize a Plotly event: the binding is tested.
    point = plot.evaluate("""el => {
        const t = el.data.find(t => t.x && t.x.length);
        const a = el._fullLayout;
        const r = el.getBoundingClientRect();
        return {x: r.left + a.xaxis._offset + a.xaxis.l2p(t.x[0]),
                y: r.top + a.yaxis._offset + a.yaxis.l2p(t.y[0])};
    }""")
    with pg.expect_response(lambda r: urlparse(r.url).path == "/api/attribute") as pending:
        pg.mouse.click(point["x"], point["y"])
    response = pending.value
    assert response.status == 200, response.url
    assert parse_qs(urlparse(response.url).query)["ds"] == ["pbmc"]
    expect(pg.locator("#cellInfo")).to_contain_text("✓")
    first_gene = pg.locator(".geneRow").first
    expect(first_gene).to_be_visible()
    gene = first_gene.get_attribute("data-gene")
    first_gene.click()
    expect(pg.locator("#geneSel")).to_have_value(gene)
    expect(pg.locator("#cipherHint")).to_contain_text(re.compile("可预测|不在"))

    # Search and choose via real controls, including their input/change events.
    def choose_gene(gene):
        with pg.expect_response(lambda r: urlparse(r.url).path == "/api/search_genes"
                                and parse_qs(urlparse(r.url).query).get("q") == [gene]) as pending:
            pg.locator("#geneFilter").fill(gene)
        assert pending.value.status == 200
        expect(pg.locator(f'#geneSel option[value="{gene}"]')).to_have_count(1)
        pg.locator("#geneSel").select_option(gene)

    choose_gene("MS4A1")
    expect(pg.locator("#cipherKoBtn")).to_be_enabled()
    expect(pg.locator("#goBtn")).to_be_enabled()

    def result_click(button, route, tab_text, status_text, ds="pbmc"):
        old_count = pg.locator(".tab").count()
        with pg.expect_response(lambda r: urlparse(r.url).path == "/api/" + route) as pending:
            pg.locator(button).click()
        response = pending.value
        assert response.status == 200, (response.url, response.status)
        assert parse_qs(urlparse(response.url).query)["ds"] == [ds], response.url
        body = response.json()
        if "dataset" in body:
            assert body["dataset"] == ds, body["dataset"]
        expect(pg.locator(".tab")).to_have_count(old_count + 1)
        expect(pg.locator(".tab.on")).to_contain_text(tab_text)
        expect(pg.locator("#status")).to_contain_text(status_text)
        expect(pg.locator(".view.on .plotbox .main-svg").first).to_be_visible()

    result_click("#exprMapBtn", "gene_expr_map", "MS4A1 表达 · pbmc", "MS4A1")
    result_click("#cipherKoBtn", "perturb_summary", "MS4A1 ko · pbmc", "上调")
    result_click("#cipherOeBtn", "perturb_summary", "MS4A1 oe · pbmc", "上调")
    result_click("#baselineBtn", "perturb_compare", "MS4A1 vs 基线 · pbmc", "基线")
    # Data flow/rendering only, not biological validity of displacement.
    panel.locator("summary").click()
    expect(pg.locator("#goBtn")).to_be_visible()
    result_click("#goBtn", "perturb", "MS4A1", "位移")
    expect(pg.locator("#status")).to_contain_text("不代表真实细胞响应")

    with pg.expect_response(lambda r: urlparse(r.url).path == "/api/data"
                            and parse_qs(urlparse(r.url).query).get("ds") == ["ms"]) as pending:
        pg.locator("#dsSel").select_option("ms")
    response = pending.value
    assert response.status == 200
    data = response.json()
    assert data["ds_key"] == "ms" and data["n_cells"] == 17799
    expect(pg.locator("#dsState")).to_contain_text("✓ MS 少突胶质细胞")
    expect(pg.locator("#cellInfo")).to_contain_text("点击右侧 UMAP")
    expect(pg.locator("#goBtn")).to_be_disabled()
    choose_gene("MOBP")
    result_click("#exprMapBtn", "gene_expr_map", "MOBP 表达 · ms", "MOBP", ds="ms")
    check_history(pg, expect)  # 切换疾病后档案仍是相同来源，不触发历史重算。


def main():
    try:
        from playwright.sync_api import sync_playwright, expect
    except ImportError:
        print("缺少 Playwright：请按 README 的独立浏览器测试说明安装；本次未执行 E2E。", file=sys.stderr)
        return 2
    artifacts = ROOT.parent / "workspace/test_artifacts/vct"
    artifacts.mkdir(exist_ok=True)
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"))
            try:
                pg = browser.new_page(viewport={"width": 1500, "height": 950})
                pg.set_default_timeout(180000)
                expect.set_options(timeout=180000)
                pg.on("pageerror", lambda e: errors.append(str(e)))
                pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                pg.on("response", lambda r: errors.append(f"HTTP {r.status} {r.url}") if r.status >= 400 else None)
                try:
                    run_checks(pg, expect)
                    assert not errors, errors[:12]
                    pg.screenshot(path=str(artifacts / "vct_verify.png"))
                except Exception:
                    try:
                        pg.screenshot(path=str(artifacts / "vct_failure.png"), timeout=10000)
                    except Exception as screenshot_error:
                        print(f"失败截图也无法保存：{screenshot_error}", file=sys.stderr)
                    raise
            finally:
                browser.close()
    except Exception as exc:
        print(f"E2E 失败：{exc}\n浏览器错误：{errors[:12]}", file=sys.stderr)
        return 1
    print("E2E 通过：真实 UMAP 点选、归因行、搜索/选基因、表达/CIPHER/基线/位移、切换 MS、独立历史面板与存档图。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
