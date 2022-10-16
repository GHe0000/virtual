# encoding = utf-8
from win32api import SetWindowLong,RGB
from win32con import WS_EX_LAYERED,WS_EX_TRANSPARENT,GWL_EXSTYLE,LWA_ALPHA
from win32gui import GetWindowLong,GetForegroundWindow,SetLayeredWindowAttributes
from keyboard import add_hotkey ,wait

class WinThrow():

    Status = False
    hWindow = None
    is_working = False
    wnd_hd_list = []

    # 设置当前窗口透明可穿透
    def setWinThrowON(self):
        hWindow = GetForegroundWindow()
        self.wnd_hd_list.append(GetForegroundWindow())
        exStyle = WS_EX_LAYERED | WS_EX_TRANSPARENT
        SetWindowLong(hWindow, GWL_EXSTYLE,exStyle)
        SetLayeredWindowAttributes(hWindow,RGB(0,0,0),150,LWA_ALPHA)

    # 设置当前窗口透明但是不可穿透
    def setWinThrowOFF(self):
        hWindow = GetForegroundWindow()
        SetWindowLong(hWindow, GWL_EXSTYLE,786704)

    # 主函数
    def main(self):
        self.is_working ^= True
        if self.is_working:
            self.setWinThrowON()
        else:
            self.setWinThrowOFF()

    # 退出时重置窗口属性
    def exitApp(self):
        for hWindow in set(self.wnd_hd_list):
            SetWindowLong(hWindow, GWL_EXSTYLE,256)

if __name__ == '__main__':
    s = WinThrow()
    add_hotkey('f2',s.main)
    wait('shift+esc')
    s.exitApp()