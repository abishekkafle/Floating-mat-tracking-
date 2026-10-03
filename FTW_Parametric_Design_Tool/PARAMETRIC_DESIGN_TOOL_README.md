# FTW Parametric Design Tool (Reconstructed)

This folder contains a transparent reconstruction of the parametric screening tool described in the manuscript **Robust Modular Floating Wetland: Design & Wave-Tank Evaluation**.

> **Provenance:** This is a reconstructed implementation based on the manuscript description. It should not be described as the original historical source code unless the original file is later recovered and verified.

## Outputs

`parametric_design_tool.py` evaluates:

- perforated-mat waterplane area and open-area fraction,
- equilibrium draft and freeboard,
- net payload capacity before deck immersion,
- approximate roll/pitch metacentric height,
- small-angle hydrostatic restoring stiffness,
- quasi-steady wind drag and overturning moment,
- service/design tether load per active line,
- optional line-strength utilization, and
- optional simplified concrete-block sliding resistance.

No third-party Python packages are required.

## Governing relations

Net waterplane area:

`A_wp = L W - N_h*pi*(d_h/2)^2`

Static draft:

`draft = total_mass / (rho_water*A_wp)`

Freeboard:

`freeboard = mat_thickness - draft`

Initial stability:

`GM = KB + BM - KG`, with `KB ≈ draft/2` and `BM = I_wp/displaced_volume`.

Because exact hole coordinates are not encoded, rectangular waterplane inertia is reduced in proportion to remaining waterplane area. This is an explicit approximation.

Wind force:

`F_w = 0.5*rho_air*C_D*A_frontal*V^2`

Wind moment:

`M_w = F_w*z_cp`

Small-angle roll restoring stiffness:

`C_phi = rho_water*g*displaced_volume*GM_roll`

Service tether tension:

`T_service = F_horizontal/(N_active*cos(theta))*k_uneven`

Design tether tension:

`T_design = T_service*k_dynamic*k_safety`

The concrete-block calculation is a simple friction screen based on estimated submerged block weight; it does not model embedment, suction, scour, cyclic degradation, or detailed soil mechanics.

## Example inputs

The provided `example_ftw_config.json` contains documented geometry/hardware where available:

- mat: 7 ft x 6 ft x 1.5 in,
- planting holes: approximately 50, 6 in diameter,
- concrete block mass: 36.5 lb,
- steel wire-rope breaking strength: 6,400 lbf.

Other values in the example (mat mass, plant/pot payload, CG, projected wind area, line angle, friction coefficient, dynamic factor, safety factor) are **illustrative placeholders**. Replace them with the actual values for each design case.

## Run

```bash
python parametric_design_tool.py example_ftw_config.json
```

Save detailed JSON output:

```bash
python parametric_design_tool.py example_ftw_config.json --json results.json
```

Run tests:

```bash
python -m unittest -v test_parametric_design_tool.py
```

## Recommended manuscript description

> A lightweight Python parametric screening tool was used during preliminary design to compare mat geometries and anchoring layouts. The tool used hydrostatic displacement to estimate draft, freeboard, payload capacity, and initial metacentric stability, and a quasi-steady aerodynamic drag relation to estimate wind loading. The environmental load was distributed among the active tether lines according to the specified line geometry and user-defined dynamic and safety factors. The implementation was intended for comparative screening rather than full hydroelastic, mooring-dynamics, or geotechnical prediction. The reconstructed source code and documented assumptions are provided in the project repository.

## Limitations

This is a low-order screening model. It does not include wave radiation/diffraction, hydroelastic plate deformation, transient mooring dynamics, cyclic fatigue, detailed current loading, or a geotechnical anchor model.
