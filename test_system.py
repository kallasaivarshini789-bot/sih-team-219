"""
Automated System Verification Script
------------------------------------
Tests:
1. Cloud removal & optical enhancement engine
2. 4-Sector CV modules (NDVI, Flood, Buildings, Land Cover)
3. Telemetry histogram & metrics generation
4. PDF report generation
"""

import os
import sys

def test_pipeline():
    print("==================================================")
    print("  RUNNING SATELLITE SYSTEM PIPELINE VERIFICATION  ")
    print("==================================================")

    from cloud_removal import CloudRemovalEngine
    from analysis_engine import AnalysisEngine
    from report_generator import generate_mission_pdf
    from sample_generator import generate_all_samples

    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, 'static', 'samples')
    results_dir = os.path.join(base_dir, 'static', 'results')
    analysis_dir = os.path.join(base_dir, 'static', 'analysis')
    reports_dir = os.path.join(base_dir, 'static', 'reports')

    for d in [samples_dir, results_dir, analysis_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n[Step 1] Ensuring sample scenes exist...")
    generate_all_samples(samples_dir)
    test_sample = os.path.join(samples_dir, 'agriculture.jpg')
    assert os.path.exists(test_sample), "Sample agriculture.jpg not found!"
    print("  -> Sample scenes verified.")

    print("\n[Step 2] Testing CloudRemovalEngine enhancement...")
    engine = CloudRemovalEngine()
    out_img = os.path.join(results_dir, 'test_cleared.png')
    out_mask = os.path.join(results_dir, 'test_mask.png')
    
    metrics = engine.enhance_image(test_sample, out_img, mask_output_path=out_mask)
    assert os.path.exists(out_img), "Cleared image was not generated!"
    assert os.path.exists(out_mask), "Cloud mask was not generated!"
    assert 'contrast_boost_pct' in metrics, "Contrast boost metric missing!"
    assert 'histogram' in metrics, "Histogram missing!"
    assert len(metrics['histogram']['before']) == 16, "Histogram before bins invalid!"
    print(f"  -> Enhancement successful! Time: {metrics['processing_time']}s, Contrast Boost: +{metrics['contrast_boost_pct']}%, Cloud Cover: {metrics['cloud_coverage_pct']}%")

    print("\n[Step 3] Testing 4 Sector CV Analysis modules...")
    cv_engine = AnalysisEngine()

    # 1. NDVI
    ndvi_out = os.path.join(analysis_dir, 'test_ndvi.png')
    ndvi_res = cv_engine.calculate_ndvi(out_img, ndvi_out)
    assert os.path.exists(ndvi_out), "NDVI map missing!"
    print(f"  -> NDVI OK! Dense Veg: {ndvi_res['dense_healthy_pct']}%, Mean NDVI: {ndvi_res['mean_ndvi_score']}")

    # 2. Flood
    flood_out = os.path.join(analysis_dir, 'test_flood.png')
    flood_res = cv_engine.detect_flood(out_img, flood_out)
    assert os.path.exists(flood_out), "Flood map missing!"
    print(f"  -> Flood OK! Water Coverage: {flood_res['water_coverage_percent']}%, Bodies: {flood_res['num_water_bodies']}")

    # 3. Buildings
    bld_out = os.path.join(analysis_dir, 'test_buildings.png')
    bld_res = cv_engine.detect_buildings(out_img, bld_out)
    assert os.path.exists(bld_out), "Building map missing!"
    print(f"  -> Buildings OK! Detected: {bld_res['buildings_detected']}, Built-up Density: {bld_res['builtup_density_percent']}%")

    # 4. Land Cover
    lc_out = os.path.join(analysis_dir, 'test_landcover.png')
    lc_res = cv_engine.classify_landcover(out_img, lc_out)
    assert os.path.exists(lc_out), "Land Cover map missing!"
    print(f"  -> Land Cover OK! Forest/Veg: {lc_res['vegetation_forest_percent']}%, Urban: {lc_res['urban_builtup_percent']}%")

    print("\n[Step 4] Testing Executive PDF Report generation...")
    report_pdf = os.path.join(reports_dir, 'test_report.pdf')
    generate_mission_pdf(
        report_path=report_pdf,
        input_img_path=test_sample,
        output_img_path=out_img,
        analysis_img_path=ndvi_out,
        metrics=metrics,
        analysis_data=ndvi_res,
        scene_name="Agricultural Test Region"
    )
    assert os.path.exists(report_pdf), "PDF report not generated!"
    print(f"  -> PDF Report OK! Size: {round(os.path.getsize(report_pdf)/1024, 1)} KB")

    print("\n==================================================")
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!    ")
    print("==================================================")

if __name__ == '__main__':
    test_pipeline()
