import json, math, unittest
from pathlib import Path
if __package__:
  from .parametric_design_tool import Geometry,MassModel,Environment,Mooring,Anchor,evaluate,wind_drag,print_report
else:
  from parametric_design_tool import Geometry,MassModel,Environment,Mooring,Anchor,evaluate,wind_drag,print_report
class Tests(unittest.TestCase):
  def test_drag(self):
    e=Environment(air_density_kg_m3=1.2,wind_speed_m_s=10,wind_drag_coefficient=1,wind_projected_area_m2=2)
    self.assertAlmostEqual(wind_drag(e),120)
  def test_draft(self):
    r=evaluate(Geometry(2,1,0.1),MassModel(10,90,0.05,0.05),Environment(water_density_kg_m3=1000),Mooring(active_lines=2))
    self.assertAlmostEqual(r['hydrostatics']['equilibrium_draft_m'],0.05)
  def test_example(self):
    c=json.loads(Path(__file__).with_name('example_ftw_config.json').read_text())
    r=evaluate(Geometry(**c['geometry']),MassModel(**c['mass']),Environment(**c['environment']),Mooring(**c['mooring']),Anchor(**c['anchor']))
    self.assertTrue(math.isfinite(r['hydrostatics']['GM_roll_m']))
    self.assertGreater(r['hydrostatics']['freeboard_m'],0)
  def test_report_when_anchor_has_no_sliding_resistance(self):
    from contextlib import redirect_stdout
    from io import StringIO
    r=evaluate(Geometry(2,1,0.1),MassModel(10,90,0.05,0.05),Environment(wind_speed_m_s=10,wind_projected_area_m2=2),Mooring(line_angle_from_horizontal_deg=80),Anchor(block_mass_kg=1,bed_friction_coefficient=0.5))
    self.assertIsNone(r['anchor']['sliding_utilization_ratio'])
    self.assertFalse(r['anchor']['passes_simple_sliding_screen'])
    output=StringIO()
    with redirect_stdout(output): print_report(r)
    self.assertIn('no sliding resistance',output.getvalue())
if __name__=='__main__': unittest.main()
