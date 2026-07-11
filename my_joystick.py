
from donkeycar.parts.controller import Joystick, JoystickController


class MyJoystick(Joystick):
    #An interface to a physical joystick available at /dev/input/js0
    def __init__(self, *args, **kwargs):
        super(MyJoystick, self).__init__(*args, **kwargs)

            
        self.button_names = {
            0x134 : 'Y',
            0x133 : 'X',
            0x130 : 'A',
            0x131 : 'B',
            0x13e : 'Right Joystick',
            0x13d : 'Left Joystick',
            0x137 : 'Right Bumper',
            0x136 : 'Left Bumper',
            0x13b : 'Start',
            0x13a : 'Select',
            0x13c : 'Home',
        }


        self.axis_names = {
            0x1 : 'Left JS Y',
            0x0 : 'Left JS X',
            0x4 : 'Right JS Y',
            0x3 : 'Right JS X',
            0x11 : 'Dpad Y',
            0x10 : 'Dpad X',
            0x5 : 'Right Trigger',
            0x2 : 'Left Trigger',
        }



class MyJoystickController(JoystickController):
    #A Controller object that maps inputs to actions
    def __init__(self, *args, **kwargs):
        super(MyJoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        #attempt to init joystick
        try:
            self.js = MyJoystick(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            print(self.dev_fn, "not found.")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        #init set of mapping from buttons to function calls
            
        self.button_down_trigger_map = {
            'A' : self.toggle_mode,
            'X' : self.erase_last_N_records,
            'Y' : self.emergency_stop,
            'Right Bumper' : self.increase_max_throttle,
            'Left Bumper' : self.decrease_max_throttle,
            'B' : self.toggle_manual_recording,
        }


        self.axis_trigger_map = {
            'Left JS X' : self.set_steering,
            'Right JS Y' : self.set_throttle,
        }


