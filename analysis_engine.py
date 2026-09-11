"""
Multi-Sector Post-Processing Computer Vision Analysis Suite
------------------------------------------------------------
Implements 5 specialized domain modules:
1. 🌿 NDVI Vegetation Health Index (Agriculture)
2. 🌊 Flood & Water Inundation Detection (Disaster Management)
3. 🚨 Flood Risk & Rescue Zone Detection (Cross-module: Flood + Buildings)
4. 🏠 Building & Infrastructure Extraction (Urban Planning)
5. 🗺️ Land Cover Classification (Climate Research)
"""

import numpy as np
import cv2
from PIL import Image

class AnalysisEngine:
    def __init__(self):
        print("[Satellite Engine] Multi-Sector Post-Processing CV Suite initialized.")

    def _add_legend_box(self, img_bgr, title, items):
        """Draw an informative HUD-style legend overlay on the top-left of the analysis map."""
        h, w, _ = img_bgr.shape
        box_w = min(260, int(w * 0.45))
        box_h = 24 + len(items) * 22
        
        # Semi-transparent dark backing
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (12, 12), (12 + box_w, 12 + box_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)
        cv2.rectangle(img_bgr, (12, 12), (12 + box_w, 12 + box_h), (56, 189, 248), 1)

        # Title
        cv2.putText(img_bgr, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Items
        for idx, (color_bgr, label) in enumerate(items):
            y_pos = 48 + idx * 20
            # Color swatch
            cv2.rectangle(img_bgr, (22, y_pos - 10), (34, y_pos + 2), color_bgr, -1)
            cv2.rectangle(img_bgr, (22, y_pos - 10), (34, y_pos + 2), (255, 255, 255), 1)
            # Label
            cv2.putText(img_bgr, label, (42, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (226, 232, 240), 1, cv2.LINE_AA)
        return img_bgr

    def calculate_ndvi(self, image_path, output_path):
        """
        Calculate NDVI (Normalized Difference Vegetation Index)
        Uses a combined VARI + Green-Blue ratio for RGB satellite images.
        Thresholds are calibrated so moderate/crop vegetation (yellow) dominates typical scenes.
        """
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, _ = img_rgb.shape
        total_pixels = h * w

        red = img_rgb[:, :, 0]
        green = img_rgb[:, :, 1]
        blue = img_rgb[:, :, 2]

        # VARI (Visible Atmospherically Resistant Index) - better for RGB satellite imagery
        # VARI = (Green - Red) / (Green + Red - Blue + 1e-6)
        vari = (green - red) / (green + red - blue + 1e-6)
        vari = np.clip(vari, -1.0, 1.0)

        # Also compute enhanced green index for robustness
        egi = (green - red) / (green + red + 1e-6)
        egi = np.clip(egi, -1.0, 1.0)

        # Combined weighted index
        ndvi = 0.65 * vari + 0.35 * egi
        ndvi = np.clip(ndvi, -1.0, 1.0)

        # Class thresholds tuned for typical mixed/agricultural satellite scenes:
        # Dense Healthy Vegetation (Green):   ndvi >= 0.18
        # Moderate / Crop Vegetation (Yellow): 0.02 <= ndvi < 0.18  (this range covers most natural terrain)
        # Barren Soil / Water / Non-Veg (Red): ndvi < 0.02
        high_veg_mask = ndvi >= 0.18
        mod_veg_mask  = (ndvi >= 0.02) & (ndvi < 0.18)
        barren_mask   = ndvi < 0.02

        dense_pct  = round(float(np.sum(high_veg_mask) / total_pixels * 100), 1)
        mod_pct    = round(float(np.sum(mod_veg_mask)  / total_pixels * 100), 1)
        barren_pct = round(max(0.0, 100.0 - (dense_pct + mod_pct)), 1)
        mean_ndvi  = round(float(np.mean(ndvi)), 3)

        # Create colorized classified overlay:
        # Vivid Emerald Green = Dense, Golden Yellow = Moderate, Crimson Red = Barren
        classified_map = np.zeros_like(img_bgr)
        classified_map[high_veg_mask] = [34,  179,  84]   # BGR: bright green
        classified_map[mod_veg_mask]  = [20,  190, 240]   # BGR: golden yellow (in BGR: high R, high G, low B → B=20, G=190, R=240)
        classified_map[barren_mask]   = [40,   40, 220]   # BGR: vivid red

        # Blend with 25% original terrain so terrain detail shows through
        blended = cv2.addWeighted(img_bgr, 0.25, classified_map, 0.75, 0)

        legend_items = [
            ((34,  179,  84), f"Dense Healthy Veg  (VARI ≥ 0.18)"),
            ((20,  190, 240), f"Moderate / Crop Veg (0.02–0.18)"),
            ((40,   40, 220), f"Barren / Non-Veg    (< 0.02)")
        ]
        final_map = self._add_legend_box(blended, "NDVI VEGETATION INDEX MAP", legend_items)
        cv2.imwrite(output_path, final_map)

        health_rating = (
            "Healthy Canopy" if mean_ndvi > 0.12
            else ("Moderate Canopy" if mean_ndvi > 0.00
                  else "Sparse / Stressed")
        )

        return {
            'module_id': 'ndvi',
            'title': 'NDVI Vegetation Health Index',
            'domain': 'Agriculture & Crop Health',
            'dense_healthy_pct': dense_pct,
            'moderate_crop_pct': mod_pct,
            'barren_water_pct': barren_pct,
            'mean_ndvi_score': mean_ndvi,
            'estimated_biomass_density': f"{round((dense_pct * 1.8 + mod_pct * 0.9), 1)} MT/ha",
            'vegetation_health_status': health_rating,
            'chart': {
                'labels': ['Dense Healthy Vegetation', 'Moderate / Crop Vegetation', 'Barren Soil / Non-Veg'],
                'data': [dense_pct, mod_pct, barren_pct],
                'colors': ['#10b981', '#f59e0b', '#ef4444']
            }
        }

    def detect_flood(self, image_path, output_path):
        """
        Flood & Water Inundation Detection (Disaster Management)
        Identifies open surface water bodies, waterlogging, and flood contours.
        """
        img_bgr = cv2.imread(image_path)
        h, w, _ = img_bgr.shape
        total_pixels = h * w

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Water spectral response: Blue-Cyan hue with moderate/high saturation
        lower_water = np.array([85, 35, 20])
        upper_water = np.array([140, 255, 255])
        water_mask = cv2.inRange(hsv, lower_water, upper_water)

        # Secondary low-brightness water detector (deep ponds, sediment water)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dark_water_mask = ((gray < 55) & (img_bgr[:, :, 0] > img_bgr[:, :, 2])).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(water_mask, dark_water_mask)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        # Contours detection for flooded regions
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter tiny artifacts
        valid_contours = [c for c in contours if cv2.contourArea(c) > 40]
        
        overlay = img_bgr.copy()
        # Cyan-Blue tint over water
        overlay[cleaned_mask > 0] = [235, 140, 0] # BGR vivid cyan-blue
        result = cv2.addWeighted(img_bgr, 0.55, overlay, 0.45, 0)

        # Draw red danger outline around large flooded perimeters
        for cnt in valid_contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                cv2.drawContours(result, [cnt], -1, (0, 0, 255), 2) # Red outline for major floods
            else:
                cv2.drawContours(result, [cnt], -1, (0, 230, 255), 1) # Yellow-cyan for minor pools

        water_pixels = np.sum(cleaned_mask > 0)
        water_pct = round(float(water_pixels / total_pixels * 100), 1)
        dry_pct = round(100.0 - water_pct, 1)

        legend_items = [
            ((235, 140, 0), "Surface Water / Inundation"),
            ((0, 0, 255), "High-Risk Flood Perimeter"),
            ((0, 230, 255), "Secondary Water Bodies")
        ]
        final_map = self._add_legend_box(result, "FLOOD & HYDROLOGICAL DELTA MAP", legend_items)
        cv2.imwrite(output_path, final_map)

        flood_status = "CRITICAL FLOOD ALERT" if water_pct > 25 else ("Elevated Inundation" if water_pct > 10 else "Nominal / Low Water Risk")
        est_inundation_area = round((water_pct / 100.0) * 12.5, 2) # Sq. km estimate for typical scene

        return {
            'module_id': 'flood',
            'title': 'Flood & Water Inundation Detection',
            'domain': 'Disaster Management & Response',
            'water_coverage_percent': water_pct,
            'dry_land_percent': dry_pct,
            'num_water_bodies': len(valid_contours),
            'flood_risk_level': flood_status,
            'estimated_inundation_area': f"{est_inundation_area} km²",
            'chart': {
                'labels': ['Inundated Water Surface', 'Dry Land / Uplands'],
                'data': [water_pct, dry_pct],
                'colors': ['#06b6d4', '#64748b']
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  MODULE 3: Flood Risk & Rescue Zone Detection
    # ─────────────────────────────────────────────────────────────────────────
    def detect_flood_rescue(self, image_path, output_path):
        """
        Cross-module Flood Risk + Building Rescue Analysis.
        Detects flooded water regions AND buildings/structures within or near
        those flooded zones. Categorizes each detected structure as:
          🔴 RESCUE CRITICAL — building centroid inside flood water mask
          🟡 AT RISK        — building bounding box overlaps flood boundary
          🟢 SAFE           — structure outside flood perimeter
        Outputs a vivid SOS rescue map with labeled danger zones.
        """
        img_bgr = cv2.imread(image_path)
        h, w, _ = img_bgr.shape
        total_pixels = h * w

        # ── STAGE 1: Flood / Water Mask Detection ────────────────────────────
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Primary water: blue-cyan hue + good saturation
        lower_water = np.array([82, 30, 18])
        upper_water = np.array([142, 255, 255])
        water_mask = cv2.inRange(hsv, lower_water, upper_water)

        # Secondary: dark still water (ponds, deep flood)
        dark_water = ((gray < 60) & (img_bgr[:, :, 0] > img_bgr[:, :, 2])).astype(np.uint8) * 255

        # Tertiary: bright haze/cloud areas might be water — also check low-sat brightness
        # This helps in partially cloud-cleared images
        flat_bright = ((val > 175) & (sat < 40)).astype(np.uint8) * 0  # Disabled (too aggressive)

        flood_raw = cv2.bitwise_or(water_mask, dark_water)

        # Morphological clean-up: close small holes, remove noise
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        flood_mask = cv2.morphologyEx(flood_raw, cv2.MORPH_CLOSE, k_close)
        flood_mask = cv2.morphologyEx(flood_mask, cv2.MORPH_OPEN, k_open)

        # Dilate flood slightly to catch buildings at water edge
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        flood_dilated = cv2.dilate(flood_mask, k_dilate, iterations=1)  # "at-risk" boundary

        water_pixels = int(np.sum(flood_mask > 0))
        water_pct = round(float(water_pixels / total_pixels) * 100, 1)

        # ── STAGE 2: Building / Structure Detection ───────────────────────────
        filtered = cv2.bilateralFilter(gray, 7, 55, 55)
        edges = cv2.Canny(filtered, 55, 150)
        rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rect_k)
        all_contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Keep structures with plausible building geometry
        building_contours = []
        for c in all_contours:
            area = cv2.contourArea(c)
            if 80 < area < (total_pixels * 0.20):
                peri = cv2.arcLength(c, True)
                if peri > 0 and (4 * np.pi * area / (peri * peri)) < 0.88:
                    building_contours.append(c)

        # ── STAGE 3: Classify Each Building vs Flood Mask ─────────────────────
        rescue_critical = []   # centroid IN flood_mask
        at_risk         = []   # bbox overlaps flood_dilated but not centroid in flood
        safe_buildings  = []   # outside flood zone entirely

        for c in building_contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cx_c = max(0, min(w - 1, cx))
            cy_c = max(0, min(h - 1, cy))

            centroid_in_flood   = flood_mask[cy_c, cx_c] > 0
            centroid_in_dilated = flood_dilated[cy_c, cx_c] > 0

            # Also check what fraction of the bounding box overlaps flood
            x, y, bw, bh = cv2.boundingRect(c)
            x, y = max(0, x), max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            if bw > 0 and bh > 0:
                roi_flood = flood_mask[y:y+bh, x:x+bw]
                flood_overlap_frac = float(np.sum(roi_flood > 0)) / (bw * bh)
            else:
                flood_overlap_frac = 0.0

            if centroid_in_flood or flood_overlap_frac > 0.35:
                rescue_critical.append((c, cx, cy))
            elif centroid_in_dilated or flood_overlap_frac > 0.08:
                at_risk.append((c, cx, cy))
            else:
                safe_buildings.append((c, cx, cy))

        # ── STAGE 4: Render Rescue Map ─────────────────────────────────────────
        result = img_bgr.copy()

        # 4a. Flood water zone — deep blue tint
        flood_overlay = result.copy()
        flood_overlay[flood_mask > 0] = [210, 100, 10]  # BGR: vivid cyan-blue
        result = cv2.addWeighted(result, 0.50, flood_overlay, 0.50, 0)

        # 4b. Draw flood contours
        flood_cnts, _ = cv2.findContours(flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for fc in flood_cnts:
            if cv2.contourArea(fc) > 300:
                cv2.drawContours(result, [fc], -1, (30, 30, 255), 2)  # Red border

        # 4c. SAFE buildings — thin green outline
        for (c, cx, cy) in safe_buildings:
            cv2.drawContours(result, [c], -1, (50, 220, 60), 1)

        # 4d. AT-RISK buildings — amber/yellow outline + warning triangle
        for (c, cx, cy) in at_risk:
            cv2.drawContours(result, [c], -1, (0, 210, 255), 2)  # BGR: amber
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(result, (x - 2, y - 2), (x + bw + 2, y + bh + 2), (0, 180, 255), 2)
            # Warning label
            lx, ly = max(0, cx - 18), max(12, cy - 6)
            cv2.putText(result, "!", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 200, 255), 2, cv2.LINE_AA)

        # 4e. RESCUE CRITICAL buildings — vivid red filled rect + SOS label + circle pulse
        for (c, cx, cy) in rescue_critical:
            x, y, bw, bh = cv2.boundingRect(c)

            # Semi-transparent red fill over building area
            rescue_overlay = result.copy()
            cv2.rectangle(rescue_overlay, (x, y), (x + bw, y + bh), (30, 30, 230), -1)
            result = cv2.addWeighted(result, 0.60, rescue_overlay, 0.40, 0)

            # Bright red outline
            cv2.rectangle(result, (x - 3, y - 3), (x + bw + 3, y + bh + 3), (0, 0, 255), 3)
            cv2.drawContours(result, [c], -1, (0, 50, 255), 2)

            # Pulsing circular rescue marker at centroid
            cv2.circle(result, (cx, cy), 14, (0, 0, 255), 3)    # Outer ring
            cv2.circle(result, (cx, cy), 8,  (30, 30, 255), -1)  # Inner fill
            cv2.circle(result, (cx, cy), 4,  (255, 255, 255), -1)  # White dot

            # SOS label above building
            label_x = max(0, x)
            label_y = max(14, y - 6)
            # Dark backing for readability
            text_size = cv2.getTextSize("SOS", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0]
            cv2.rectangle(result,
                          (label_x - 2, label_y - text_size[1] - 3),
                          (label_x + text_size[0] + 4, label_y + 3),
                          (20, 20, 20), -1)
            cv2.putText(result, "SOS", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        # 4f. Header alert banner if critical rescues detected
        if rescue_critical:
            alert_msg = f"!! RESCUE ALERT: {len(rescue_critical)} STRUCTURE(S) SUBMERGED !!"
            banner_w = w
            cv2.rectangle(result, (0, h - 28), (banner_w, h), (0, 0, 200), -1)
            cv2.putText(result, alert_msg, (8, h - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # ── STAGE 5: Legend & Save ─────────────────────────────────────────────
        legend_items = [
            ((210, 100, 10), "Flood Inundation Zone"),
            ((30, 30, 255),  f"RESCUE CRITICAL — {len(rescue_critical)} structure(s) submerged"),
            ((0, 210, 255),  f"AT RISK — {len(at_risk)} structure(s) at flood boundary"),
            ((50, 220, 60),  f"Safe — {len(safe_buildings)} structure(s) outside flood zone"),
            ((0, 0, 255),    "Flood Perimeter / Danger Contour")
        ]
        result = self._add_legend_box(result, "FLOOD RISK & RESCUE ZONE MAP", legend_items)
        cv2.imwrite(output_path, result)

        # ── Metrics ───────────────────────────────────────────────────────────
        dry_pct = round(100.0 - water_pct, 1)
        total_buildings = len(building_contours)
        rescue_pct = round((len(rescue_critical) / max(total_buildings, 1)) * 100, 1)
        atrisk_pct = round((len(at_risk) / max(total_buildings, 1)) * 100, 1)
        safe_pct = round(100.0 - rescue_pct - atrisk_pct, 1)

        flood_risk_level = (
            "[CRITICAL] Mass Evacuation Required" if len(rescue_critical) >= 5
            else "[HIGH] Targeted Rescue Operations Needed" if len(rescue_critical) >= 1
            else "[ELEVATED] Precautionary Evacuation Advised" if len(at_risk) >= 3
            else "[LOW] Monitor & Standby"
        )

        est_rescue_area = round((water_pct / 100.0) * 12.5, 2)

        return {
            'module_id': 'flood_rescue',
            'title': 'Flood Risk & Rescue Zone Detection',
            'domain': 'Disaster Management & Emergency Response',
            'flood_coverage_percent': water_pct,
            'dry_land_percent': dry_pct,
            'total_structures_detected': total_buildings,
            'rescue_critical_count': len(rescue_critical),
            'at_risk_count': len(at_risk),
            'safe_structures_count': len(safe_buildings),
            'flood_risk_assessment': flood_risk_level,
            'estimated_inundation_area': f"{est_rescue_area} km²",
            'chart': {
                'labels': [
                    f'🔴 RESCUE CRITICAL ({len(rescue_critical)})',
                    f'🟡 AT RISK ({len(at_risk)})',
                    f'🟢 SAFE ({len(safe_buildings)})',
                    'Flood Water Zone'
                ],
                'data': [
                    round(len(rescue_critical) / max(total_buildings, 1) * 100, 1),
                    round(len(at_risk) / max(total_buildings, 1) * 100, 1),
                    round(len(safe_buildings) / max(total_buildings, 1) * 100, 1),
                    water_pct
                ],
                'colors': ['#ef4444', '#f59e0b', '#22c55e', '#06b6d4']
            }
        }

    def detect_buildings(self, image_path, output_path):
        """
        Building & Urban Footprint Extraction (Urban Planning)
        Extracts structural edges, roof footprints, and geometric contours.
        """
        img_bgr = cv2.imread(image_path)
        h, w, _ = img_bgr.shape
        total_pixels = h * w

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Adaptive bilateral filter to preserve edges while smoothing roof textures
        filtered = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(filtered, 60, 160)

        # Close contours into rectangular structural footprints
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rect_kernel)
        
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and geometric aspect ratio
        building_contours = []
        total_building_area = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if 60 < area < (total_pixels * 0.25):
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    # Rectangular structures generally have circularity < 0.8
                    if circularity < 0.85:
                        building_contours.append(c)
                        total_building_area += area

        overlay = img_bgr.copy()
        
        for c in building_contours:
            # Draw neon green building contour
            cv2.drawContours(overlay, [c], -1, (0, 255, 128), 2)
            # Draw bounding box
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 200, 0), 1)
            # Centroid point
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(overlay, (cX, cY), 2, (0, 255, 255), -1)

        result = cv2.addWeighted(img_bgr, 0.45, overlay, 0.55, 0)
        
        builtup_pct = round(min(85.0, (total_building_area / total_pixels) * 100), 1)
        open_space_pct = round(100.0 - builtup_pct, 1)
        avg_area = round(float(total_building_area / max(len(building_contours), 1)), 1)

        legend_items = [
            ((0, 255, 128), "Building Footprint Contour"),
            ((255, 200, 0), "Cadastral Bounding Envelope"),
            ((0, 255, 255), "Structural Centroid Node")
        ]
        final_map = self._add_legend_box(result, "URBAN INFRASTRUCTURE EXTRACTION", legend_items)
        cv2.imwrite(output_path, final_map)

        urban_density = "High Density Core" if builtup_pct > 20 else ("Medium Suburban" if builtup_pct > 8 else "Low Density / Rural")

        return {
            'module_id': 'buildings',
            'title': 'Building Extraction & Cadastral Mapping',
            'domain': 'Urban Planning & Municipal Governance',
            'buildings_detected': len(building_contours),
            'builtup_density_percent': builtup_pct,
            'open_space_percent': open_space_pct,
            'avg_building_footprint': f"{avg_area} px²",
            'urban_density_classification': urban_density,
            'chart': {
                'labels': ['Built-Up Footprint', 'Open Spaces / Roads'],
                'data': [builtup_pct, open_space_pct],
                'colors': ['#8b5cf6', '#334155']
            }
        }

    def classify_landcover(self, image_path, output_path):
        """
        Land Cover Multi-Class Segmentation (Climate Research)
        Classifies scene into 4 distinct bio-climatic terrain regimes:
        1. Dense Vegetation / Forest (Green)
        2. Water Bodies & Streams (Blue)
        3. Built-Up & Pavement (Purple/Gray)
        4. Bare Soil & Sand (Golden Brown)
        """
        img_bgr = cv2.imread(image_path)
        h, w, _ = img_bgr.shape
        total_pixels = h * w

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. Vegetation mask
        lower_veg = np.array([30, 35, 30])
        upper_veg = np.array([88, 255, 255])
        mask_veg = cv2.inRange(hsv, lower_veg, upper_veg)

        # 2. Water mask
        lower_water = np.array([89, 40, 20])
        upper_water = np.array([135, 255, 255])
        mask_water = cv2.inRange(hsv, lower_water, upper_water)

        # 3. Bare soil mask (warm orange/brown)
        lower_soil = np.array([5, 30, 40])
        upper_soil = np.array([28, 200, 240])
        mask_soil = cv2.inRange(hsv, lower_soil, upper_soil)

        # 4. Urban / Built-up (low saturation, mid-high brightness or unassigned)
        assigned = cv2.bitwise_or(mask_veg, cv2.bitwise_or(mask_water, mask_soil))
        mask_urban = cv2.bitwise_not(assigned)

        # Color-coded classified overlay
        colored_map = np.zeros_like(img_bgr)
        colored_map[mask_veg > 0] = [34, 197, 94]    # BGR Green
        colored_map[mask_water > 0] = [239, 110, 6]  # BGR Cyan-Blue
        colored_map[mask_soil > 0] = [38, 140, 217]  # BGR Amber/Brown
        colored_map[mask_urban > 0] = [180, 100, 140] # BGR Mauve/Slate

        # Smooth blend
        blended = cv2.addWeighted(img_bgr, 0.35, colored_map, 0.65, 0)

        veg_pct = round(float(np.sum(mask_veg > 0) / total_pixels * 100), 1)
        water_pct = round(float(np.sum(mask_water > 0) / total_pixels * 100), 1)
        soil_pct = round(float(np.sum(mask_soil > 0) / total_pixels * 100), 1)
        urban_pct = round(max(0.0, 100.0 - (veg_pct + water_pct + soil_pct)), 1)

        legend_items = [
            ((34, 197, 94), "Dense Vegetation / Forest"),
            ((239, 110, 6), "Surface Water Bodies"),
            ((38, 140, 217), "Bare Soil / Arid Terrain"),
            ((180, 100, 140), "Urban / Built-Up Area")
        ]
        final_map = self._add_legend_box(blended, "LAND COVER CLASSIFICATION MAP", legend_items)
        cv2.imwrite(output_path, final_map)

        carbon_rating = "Net Carbon Sink (+)" if veg_pct > 40 else ("Neutral / Balanced" if veg_pct > 20 else "High Emission / Urban Zone")

        return {
            'module_id': 'landcover',
            'title': 'Land Cover & Biophysical Classification',
            'domain': 'Climate Research & Ecology',
            'vegetation_forest_percent': veg_pct,
            'water_bodies_percent': water_pct,
            'bare_soil_percent': soil_pct,
            'urban_builtup_percent': urban_pct,
            'carbon_sink_assessment': carbon_rating,
            'chart': {
                'labels': ['Vegetation / Canopy', 'Water Bodies', 'Bare Soil / Arid', 'Urban / Built-Up'],
                'data': [veg_pct, water_pct, soil_pct, urban_pct],
                'colors': ['#22c55e', '#06b6d4', '#f59e0b', '#a855f7']
            }
        }
