"""
Quick viewer script — press arrow keys to control velocity.
  ↑/↓ : +0.1 / -0.1 m/s forward velocity
  ←/→ : +0.1 / -0.1 rad/s yaw rate
  Ctrl : zero all commands
Close the viewer window to stop.
"""
import sys
sys.path.insert(0, 'src')
from main import QuadrupedController

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=30.0, help='Simulation duration (s)')
    p.add_argument('--gait', default='trot', choices=['trot', 'walk', 'bound', 'stand'])
    p.add_argument('--vx', type=float, default=0.3, help='Forward velocity [m/s]')
    args = p.parse_args()

    ctrl = QuadrupedController('models/a2_scene.xml')

    # Set command based on gait type
    if args.gait == 'stand':
        # Stand gait: zero velocity command
        ctrl.set_command(0.0, 0.0, 0.0)
    else:
        # Locomotion gait: use vx command
        ctrl.set_command(args.vx, 0.0, 0.0)

    print(f"Gait: {args.gait} | cmd: vx={args.vx}")
    print("Controls: ↑↓←→ adjust velocity, Ctrl=stop, close window=quit\n")
    ctrl.run_simulation(duration=args.duration, render=True, debug=False)
