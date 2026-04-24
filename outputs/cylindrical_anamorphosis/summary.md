# Cylindrical-mirror anamorphosis batch

This batch uses the inverse chain `I -> H -> A`: line-cylinder intersection on `VI`, cylinder normal `n`, specular reflection `d_out = d_in - 2(d_in·n)n`, then line-plane intersection with `z=0`.

## Selected common geometry

- Cylinder center `(x0, y0)=(0.0, 40.0) mm`, radius `R=22.0 mm`, height `H=180.0 mm`.
- Viewpoint `V=(0.0, -200.0, 140.0) mm`. Virtual image plane `y=75.0 mm`, `z` center `12.0 mm`.
- A4 paper is modeled as `210 x 297 mm`; the cylinder footprint is clipped with an extra `3.0 mm` clearance.

## Per-image outputs

- **fig3**: virtual plane `22.82 x 34.00 mm`, A4-valid fraction `0.999`, paper bbox `210.0 x 102.2 mm`.
  - pattern: `/home/xianz/huazhongbei/outputs/cylindrical_anamorphosis/fig3_paper_pattern.png`
  - diagnostic: `/home/xianz/huazhongbei/outputs/cylindrical_anamorphosis/fig3_diagnostic.png`
  - occupancy: `/home/xianz/huazhongbei/outputs/cylindrical_anamorphosis/fig3_occupancy.png`
- **fig4**: virtual plane `51.32 x 28.00 mm`, A4-valid fraction `0.893`, paper bbox `210.0 x 238.1 mm`.
  - pattern: `/home/xianz/huazhongbei/outputs/cylindrical_anamorphosis/fig4_paper_pattern.png`
  - diagnostic: `/home/xianz/huazhongbei/outputs/cylindrical_anamorphosis/fig4_diagnostic.png`
  - occupancy: `/home/xianz/huazhongbei/outputs/cylindrical_anamorphosis/fig4_occupancy.png`

## Rerun

```bash
python /home/xianz/huazhongbei/anamorphosis/generate_patterns.py
```
