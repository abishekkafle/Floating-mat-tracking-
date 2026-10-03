#!/usr/bin/env python3
"""Reconstructed FTW parametric design screening tool.

This implementation is reconstructed from the manuscript description; it is not
claimed to be the original historical source code.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

G = 9.80665

@dataclass
class Geometry:
    length_m: float
    width_m: float
    thickness_m: float
    hole_count: int = 0
    hole_diameter_m: float = 0.0
    @property
    def gross_area(self): return self.length_m*self.width_m
    @property
    def hole_area(self): return self.hole_count*math.pi*(self.hole_diameter_m/2)**2
    @property
    def net_area(self): return self.gross_area-self.hole_area
    @property
    def solid_volume(self): return self.net_area*self.thickness_m
    def validate(self):
        if min(self.length_m,self.width_m,self.thickness_m)<=0: raise ValueError('Mat dimensions must be positive.')
        if self.hole_count<0: raise ValueError('hole_count must be nonnegative.')
        if self.hole_count and self.hole_diameter_m<=0: raise ValueError('hole_diameter_m must be positive.')
        if self.net_area<=0: raise ValueError('Planting holes remove all/equivalent plan area.')

@dataclass
class MassModel:
    mat_mass_kg: float
    payload_mass_kg: float
    mat_cg_from_bottom_m: Optional[float] = None
    payload_cg_from_bottom_m: float = 0.0
    def total(self): return self.mat_mass_kg+self.payload_mass_kg
    def kg(self,t):
        matkg=t/2 if self.mat_cg_from_bottom_m is None else self.mat_cg_from_bottom_m
        return (self.mat_mass_kg*matkg+self.payload_mass_kg*self.payload_cg_from_bottom_m)/self.total()
    def validate(self):
        if self.mat_mass_kg<0 or self.payload_mass_kg<0 or self.total()<=0: raise ValueError('Masses must be nonnegative and total mass > 0.')

@dataclass
class Environment:
    water_density_kg_m3: float = 997.0
    air_density_kg_m3: float = 1.225
    wind_speed_m_s: float = 0.0
    wind_drag_coefficient: float = 1.2
    wind_projected_area_m2: float = 0.0
    wind_center_of_pressure_above_waterline_m: float = 0.0
    other_horizontal_load_n: float = 0.0

@dataclass
class Mooring:
    active_lines: int = 1
    line_angle_from_horizontal_deg: float = 0.0
    dynamic_factor: float = 1.0
    safety_factor: float = 1.0
    uneven_load_factor: float = 1.0
    line_breaking_strength_n: Optional[float] = None

@dataclass
class Anchor:
    block_mass_kg: Optional[float] = None
    concrete_density_kg_m3: float = 2300.0
    bed_friction_coefficient: Optional[float] = None


def wind_drag(env):
    return 0.5*env.air_density_kg_m3*env.wind_drag_coefficient*env.wind_projected_area_m2*env.wind_speed_m_s**2


def evaluate(g:Geometry,m:MassModel,e:Environment,mo:Mooring,a:Optional[Anchor]=None)->Dict[str,Any]:
    g.validate(); m.validate()
    if e.water_density_kg_m3<=0 or e.air_density_kg_m3<=0: raise ValueError('Densities must be positive.')
    if mo.active_lines<=0 or not (0<=mo.line_angle_from_horizontal_deg<90): raise ValueError('Invalid mooring geometry.')
    total=m.total(); draft=total/(e.water_density_kg_m3*g.net_area); freeboard=g.thickness_m-draft
    max_disp_mass=e.water_density_kg_m3*g.solid_volume
    net_payload=max_disp_mass-m.mat_mass_kg; payload_margin=net_payload-m.payload_mass_kg
    disp_vol=total/e.water_density_kg_m3; kb=draft/2; kg=m.kg(g.thickness_m)
    area_ratio=g.net_area/g.gross_area
    iroll=(g.length_m*g.width_m**3/12)*area_ratio
    ipitch=(g.width_m*g.length_m**3/12)*area_ratio
    bmroll=iroll/disp_vol; bmpitch=ipitch/disp_vol
    gmroll=kb+bmroll-kg; gmpitch=kb+bmpitch-kg
    wind=wind_drag(e); moment=wind*e.wind_center_of_pressure_above_waterline_m
    croll=e.water_density_kg_m3*G*disp_vol*gmroll
    heel_deg=math.degrees(moment/croll) if croll>0 else None
    theta=math.radians(mo.line_angle_from_horizontal_deg)
    service=(wind+e.other_horizontal_load_n)/mo.active_lines/math.cos(theta)*mo.uneven_load_factor
    design=service*mo.dynamic_factor*mo.safety_factor
    util=design/mo.line_breaking_strength_n if mo.line_breaking_strength_n else None
    anchor_out={'evaluated':False,'note':'Simplified friction screen only; excludes embedment, suction, scour and cyclic soil behavior.'}
    if a and a.block_mass_kg and a.bed_friction_coefficient:
        vconc=a.block_mass_kg/a.concrete_density_kg_m3
        wsub=a.block_mass_kg*G-e.water_density_kg_m3*G*vconc
        vert=design*math.sin(theta); horiz=design*math.cos(theta)
        normal=max(0,wsub-vert); resist=a.bed_friction_coefficient*normal
        au=horiz/resist if resist>0 else None
        anchor_out.update({'evaluated':True,'estimated_submerged_weight_n':wsub,'sliding_resistance_n':resist,'horizontal_design_component_n':horiz,'sliding_utilization_ratio':au,'passes_simple_sliding_screen':bool(au is not None and au<=1)})
    warnings=[]
    if freeboard<0: warnings.append('Calculated draft exceeds mat thickness.')
    if gmroll<=0 or gmpitch<=0: warnings.append('A metacentric height is non-positive.')
    if util is not None and util>1: warnings.append('Design line tension exceeds supplied breaking strength.')
    return {
      'inputs':{'geometry':asdict(g),'mass':asdict(m),'environment':asdict(e),'mooring':asdict(mo),'anchor':asdict(a) if a else None},
      'geometry':{'gross_plan_area_m2':g.gross_area,'total_hole_area_m2':g.hole_area,'net_waterplane_area_m2':g.net_area,'open_area_fraction':g.hole_area/g.gross_area},
      'hydrostatics':{'total_operating_mass_kg':total,'equilibrium_draft_m':draft,'freeboard_m':freeboard,'net_payload_capacity_kg':net_payload,'payload_margin_kg':payload_margin,'combined_KG_m':kg,'GM_roll_m':gmroll,'GM_pitch_m':gmpitch,'roll_restoring_stiffness_Nm_per_rad':croll},
      'wind':{'wind_drag_force_n':wind,'wind_overturning_moment_nm':moment,'small_angle_roll_heel_deg':heel_deg},
      'mooring':{'service_tension_per_active_line_n':service,'design_tension_per_active_line_n':design,'line_breaking_strength_n':mo.line_breaking_strength_n,'line_utilization_ratio':util},
      'anchor':anchor_out,'warnings':warnings}


def load(path):
    c=json.loads(Path(path).read_text())
    return Geometry(**c['geometry']),MassModel(**c['mass']),Environment(**c.get('environment',{})),Mooring(**c.get('mooring',{})),Anchor(**c['anchor']) if c.get('anchor') else None


def print_report(r):
    h=r['hydrostatics']; w=r['wind']; mo=r['mooring']; geo=r['geometry']
    print('\nFTW PARAMETRIC DESIGN SCREEN\n'+'='*64)
    print(f"Net waterplane area       : {geo['net_waterplane_area_m2']:.3f} m^2")
    print(f"Open-area fraction        : {100*geo['open_area_fraction']:.1f} %")
    print(f"Draft / freeboard         : {1000*h['equilibrium_draft_m']:.1f} / {1000*h['freeboard_m']:.1f} mm")
    print(f"Net payload capacity      : {h['net_payload_capacity_kg']:.1f} kg")
    print(f"GM roll / pitch           : {h['GM_roll_m']:.3f} / {h['GM_pitch_m']:.3f} m")
    print(f"Wind drag                 : {w['wind_drag_force_n']:.1f} N")
    print(f"Wind overturning moment   : {w['wind_overturning_moment_nm']:.1f} N m")
    print(f"Service/design line load  : {mo['service_tension_per_active_line_n']:.1f} / {mo['design_tension_per_active_line_n']:.1f} N")
    if mo['line_utilization_ratio'] is not None: print(f"Line utilization          : {100*mo['line_utilization_ratio']:.2f} %")
    if r['anchor'].get('evaluated'):
        ratio = r['anchor']['sliding_utilization_ratio']
        print(f"Anchor sliding utilization: {ratio:.3f}" if ratio is not None else 'Anchor sliding utilization: unavailable (no sliding resistance)')
    if r['warnings']:
        print('Warnings:'); [print(' - '+x) for x in r['warnings']]


def main():
    p=argparse.ArgumentParser(description='Reconstructed FTW hydrostatic/aerodynamic screening tool.')
    p.add_argument('config'); p.add_argument('--json',dest='json_path')
    a=p.parse_args(); r=evaluate(*load(a.config)); print_report(r)
    if a.json_path: Path(a.json_path).write_text(json.dumps(r,indent=2))

if __name__=='__main__': main()
