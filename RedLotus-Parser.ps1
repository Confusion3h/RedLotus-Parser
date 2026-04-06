Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

if (-not ([System.Management.Automation.PSTypeName]'InputSimulator').Type) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class WinApi {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
    [DllImport("user32.dll")]
    public static extern bool GetCursorInfo(out CURSORINFO pci);

    [StructLayout(LayoutKind.Sequential)]
    public struct CURSORINFO {
        public Int32 cbSize;
        public Int32 flags;
        public IntPtr hCursor;
        public POINT ptScreenPos;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct POINT {
        public Int32 x;
        public Int32 y;
    }
}

public class InputSimulator {
    [DllImport("user32.dll")]
    static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);

    [StructLayout(LayoutKind.Sequential)]
    struct INPUT {
        public uint type;
        public MOUSEINPUT mi;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct MOUSEINPUT {
        public int dx;
        public int dy;
        public uint mouseData;
        public uint dwFlags;
        public uint time;
        public IntPtr dwExtraInfo;
    }

    const uint INPUT_MOUSE = 0;
    const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
    const uint MOUSEEVENTF_LEFTUP = 0x0004;
    const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
    const uint MOUSEEVENTF_RIGHTUP = 0x0010;

    private static Random rand = new Random();

    public static double NextGaussian(double mean, double stdDev) {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }

     public static void LeftMouseDown() {
         INPUT[] down = new INPUT[1];
         down[0].type = INPUT_MOUSE;
         down[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
         SendInput(1, down, Marshal.SizeOf(typeof(INPUT)));
     }

     public static void LeftMouseUp() {
         INPUT[] up = new INPUT[1];
         up[0].type = INPUT_MOUSE;
         up[0].mi.dwFlags = MOUSEEVENTF_LEFTUP;
         SendInput(1, up, Marshal.SizeOf(typeof(INPUT)));
     }

     public static void RightMouseDown() {
         INPUT[] down = new INPUT[1];
         down[0].type = INPUT_MOUSE;
         down[0].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
         SendInput(1, down, Marshal.SizeOf(typeof(INPUT)));
     }

     public static void RightMouseUp() {
         INPUT[] up = new INPUT[1];
         up[0].type = INPUT_MOUSE;
         up[0].mi.dwFlags = MOUSEEVENTF_RIGHTUP;
         SendInput(1, up, Marshal.SizeOf(typeof(INPUT)));
     }

    public static void BlockHit() {
        INPUT[] input = new INPUT[2];
        input[0].type = INPUT_MOUSE;
        input[0].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
        input[1].type = INPUT_MOUSE;
        input[1].mi.dwFlags = MOUSEEVENTF_RIGHTUP;
        SendInput(2, input, Marshal.SizeOf(typeof(INPUT)));
    }
}

public class GlobalHotkey {
    [DllImport("user32.dll")]
    public static extern short GetAsyncKeyState(int vKey);

    public static bool IsKeyPressed(int vKey) {
        return (GetAsyncKeyState(vKey) & 0x8000) != 0;
    }
}
"@
}

# --- State ---
$script:leftBindKey = 0
$script:rightBindKey = 0
$script:leftEnabled = $false
$script:rightEnabled = $false
$script:hideKey = 0x75  # F6
$script:leftCPS = 12
$script:rightCPS = 12
$script:rand = New-Object System.Random
$script:isStealth = $false
$script:isBlatant = $false
$script:isBlockHit = $false
$script:onlyInGame = $true
$script:capturingLeftKey = $false
$script:capturingRightKey = $false
$script:lastBlockHit = 0

$script:keyMap = @{
    'F1' = 0x70; 'F2' = 0x71; 'F3' = 0x72; 'F4' = 0x73; 'F5' = 0x74; 'F6' = 0x75
    'F7' = 0x76; 'F8' = 0x77; 'F9' = 0x78; 'F10' = 0x79; 'F11' = 0x7A; 'F12' = 0x7B
    'A' = 0x41; 'B' = 0x42; 'C' = 0x43; 'D' = 0x44; 'E' = 0x45; 'F' = 0x46
    'G' = 0x47; 'H' = 0x48; 'I' = 0x49; 'J' = 0x4A; 'K' = 0x4B; 'L' = 0x4C
    'M' = 0x4D; 'N' = 0x4E; 'O' = 0x4F; 'P' = 0x50; 'Q' = 0x51; 'R' = 0x52
    'S' = 0x53; 'T' = 0x54; 'U' = 0x55; 'V' = 0x56; 'W' = 0x57; 'X' = 0x58
    'Y' = 0x59; 'Z' = 0x5A; 'D0' = 0x30; 'D1' = 0x31; 'D2' = 0x32; 'D3' = 0x33
    'D4' = 0x34; 'D5' = 0x35; 'D6' = 0x36; 'D7' = 0x37; 'D8' = 0x38; 'D9' = 0x39
    'Space' = 0x20; 'Shift' = 0x10; 'Alt' = 0x12
}

# --- UI ---
$form = New-Object System.Windows.Forms.Form
$form.Text = "PrivateClicker v9.0"
$form.Size = New-Object System.Drawing.Size(350, 520)
$form.FormBorderStyle = "None"
$form.BackColor = [System.Drawing.Color]::FromArgb(10, 10, 15)
$form.StartPosition = "CenterScreen"
$form.TopMost = $false

$path = New-Object System.Drawing.Drawing2D.GraphicsPath
$rect = New-Object System.Drawing.Rectangle(0, 0, 350, 520)
$radius = 20
$path.AddArc($rect.X, $rect.Y, $radius, $radius, 180, 90)
$path.AddArc($rect.Right - $radius, $rect.Y, $radius, $radius, 270, 90)
$path.AddArc($rect.Right - $radius, $rect.Bottom - $radius, $radius, $radius, 0, 90)
$path.AddArc($rect.X, $rect.Bottom - $radius, $radius, $radius, 90, 90)
$path.CloseFigure()
$form.Region = New-Object System.Drawing.Region($path)

$titleBar = New-Object System.Windows.Forms.Panel
$titleBar.Size = New-Object System.Drawing.Size(350, 45)
$titleBar.BackColor = [System.Drawing.Color]::FromArgb(20, 20, 25)
$form.Controls.Add($titleBar)

$script:isDragging = $false; $script:dragOffset = New-Object System.Drawing.Point(0, 0)
$titleBar.Add_MouseDown({ if ($_.Button -eq "Left") { $script:isDragging = $true; $script:dragOffset = $_.Location } })
$titleBar.Add_MouseMove({
    if ($script:isDragging) {
        $curPos = [System.Windows.Forms.Control]::MousePosition
        $newX = [int]$curPos.X - [int]$script:dragOffset.X
        $newY = [int]$curPos.Y - [int]$script:dragOffset.Y
        $form.Location = New-Object System.Drawing.Point($newX, $newY)
    }
})
$titleBar.Add_MouseUp({ $script:isDragging = $false })

$labelTitle = New-Object System.Windows.Forms.Label
$labelTitle.Text = "PRIVATE CLICKER 9.0"
$labelTitle.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
$labelTitle.ForeColor = [System.Drawing.Color]::Cyan
$labelTitle.Location = New-Object System.Drawing.Point(15, 12)
$labelTitle.Size = New-Object System.Drawing.Size(200, 25)
$titleBar.Controls.Add($labelTitle)

$btnClose = New-Object System.Windows.Forms.Button
$btnClose.Text = "X"; $btnClose.Size = New-Object System.Drawing.Size(30, 30); $btnClose.Location = New-Object System.Drawing.Point(310, 8)
$btnClose.FlatStyle = "Flat"; $btnClose.FlatAppearance.BorderSize = 0; $btnClose.ForeColor = [System.Drawing.Color]::White
$btnClose.Add_Click({ $form.Close() })
$titleBar.Controls.Add($btnClose)

$container = New-Object System.Windows.Forms.Panel
$container.Size = New-Object System.Drawing.Size(330, 455)
$container.Location = New-Object System.Drawing.Point(10, 55)
$container.BackColor = [System.Drawing.Color]::FromArgb(25, 25, 30)
$form.Controls.Add($container)

$labelL = New-Object System.Windows.Forms.Label; $labelL.Text = "LEFT BIND (ENABLE)"; $labelL.ForeColor = [System.Drawing.Color]::White; $labelL.Location = New-Object System.Drawing.Point(20, 20); $container.Controls.Add($labelL)
$btnL = New-Object System.Windows.Forms.Button; $btnL.Text = "SET"; $btnL.Location = New-Object System.Drawing.Point(240, 18); $btnL.Size = New-Object System.Drawing.Size(70, 25); $btnL.FlatStyle = "Flat"; $btnL.FlatAppearance.BorderColor = [System.Drawing.Color]::Cyan; $btnL.ForeColor = [System.Drawing.Color]::Cyan
$btnL.Add_Click({ $btnL.Text = "..."; $script:capturingLeftKey = $true })
$container.Controls.Add($btnL)
$barL = New-Object System.Windows.Forms.TrackBar; $barL.Minimum = 6; $barL.Maximum = 20; $barL.Value = 12; $barL.Location = New-Object System.Drawing.Point(20, 50); $barL.Size = New-Object System.Drawing.Size(290, 45)
$barL.Add_Scroll({ $script:leftCPS = $barL.Value; $labelLVal.Text = "$($barL.Value) CPS" })
$container.Controls.Add($barL)
$labelLVal = New-Object System.Windows.Forms.Label; $labelLVal.Text = "12 CPS"; $labelLVal.ForeColor = [System.Drawing.Color]::Gray; $labelLVal.Location = New-Object System.Drawing.Point(260, 85); $container.Controls.Add($labelLVal)

$labelR = New-Object System.Windows.Forms.Label; $labelR.Text = "RIGHT BIND (ENABLE)"; $labelR.ForeColor = [System.Drawing.Color]::White; $labelR.Location = New-Object System.Drawing.Point(20, 130); $container.Controls.Add($labelR)
$btnR = New-Object System.Windows.Forms.Button; $btnR.Text = "SET"; $btnR.Location = New-Object System.Drawing.Point(240, 128); $btnR.Size = New-Object System.Drawing.Size(70, 25); $btnR.FlatStyle = "Flat"; $btnR.FlatAppearance.BorderColor = [System.Drawing.Color]::Cyan; $btnR.ForeColor = [System.Drawing.Color]::Cyan
$btnR.Add_Click({ $btnR.Text = "..."; $script:capturingRightKey = $true })
$container.Controls.Add($btnR)
$barR = New-Object System.Windows.Forms.TrackBar; $barR.Minimum = 6; $barR.Maximum = 20; $barR.Value = 12; $barR.Location = New-Object System.Drawing.Point(20, 160); $barR.Size = New-Object System.Drawing.Size(290, 45)
$barR.Add_Scroll({ $script:rightCPS = $barR.Value; $labelRVal.Text = "$($barR.Value) CPS" })
$container.Controls.Add($barR)
$labelRVal = New-Object System.Windows.Forms.Label; $labelRVal.Text = "12 CPS"; $labelRVal.ForeColor = [System.Drawing.Color]::Gray; $labelRVal.Location = New-Object System.Drawing.Point(260, 195); $container.Controls.Add($labelRVal)

$checkBlatant = New-Object System.Windows.Forms.CheckBox; $checkBlatant.Text = "BLATANT MODE (VULCAN)"; $checkBlatant.ForeColor = [System.Drawing.Color]::Red; $checkBlatant.Location = New-Object System.Drawing.Point(20, 250); $checkBlatant.Size = New-Object System.Drawing.Size(290, 25)
$checkBlatant.Add_CheckedChanged({ $script:isBlatant = $checkBlatant.Checked })
$container.Controls.Add($checkBlatant)

$checkOnlyGame = New-Object System.Windows.Forms.CheckBox; $checkOnlyGame.Text = "ONLY IN GAME FOCUS"; $checkOnlyGame.Checked = $true; $checkOnlyGame.ForeColor = [System.Drawing.Color]::White; $checkOnlyGame.Location = New-Object System.Drawing.Point(20, 280); $checkOnlyGame.Size = New-Object System.Drawing.Size(290, 25)
$checkOnlyGame.Add_CheckedChanged({ $script:onlyInGame = $checkOnlyGame.Checked })
$container.Controls.Add($checkOnlyGame)

$checkBlockHit = New-Object System.Windows.Forms.CheckBox; $checkBlockHit.Text = "AUTO BLOCK-HIT"; $checkBlockHit.ForeColor = [System.Drawing.Color]::Cyan; $checkBlockHit.Location = New-Object System.Drawing.Point(20, 310); $checkBlockHit.Size = New-Object System.Drawing.Size(290, 25)
$checkBlockHit.Add_CheckedChanged({ $script:isBlockHit = $checkBlockHit.Checked })
$container.Controls.Add($checkBlockHit)

$labelStatus = New-Object System.Windows.Forms.Label; $labelStatus.Text = "STATUS: L: OFF | R: OFF"; $labelStatus.ForeColor = [System.Drawing.Color]::Gray; $labelStatus.Location = New-Object System.Drawing.Point(20, 360); $labelStatus.Size = New-Object System.Drawing.Size(290, 20); $container.Controls.Add($labelStatus)

$labelInfo = New-Object System.Windows.Forms.Label; $labelInfo.Text = "BIND = TOGGLE | F6 = HIDE"; $labelInfo.ForeColor = [System.Drawing.Color]::Cyan; $labelInfo.Font = New-Object System.Drawing.Font("Segoe UI", 8, [System.Drawing.FontStyle]::Italic); $labelInfo.Location = New-Object System.Drawing.Point(20, 400); $labelInfo.Size = New-Object System.Drawing.Size(290, 40); $container.Controls.Add($labelInfo)

# --- ENGINE ---
function Get-IsFocused {
    if (-not $script:onlyInGame) { return $true }
    $sb = New-Object System.Text.StringBuilder 256
    $hWnd = [WinApi]::GetForegroundWindow()
    [WinApi]::GetWindowText($hWnd, $sb, $sb.Capacity)
    $title = $sb.ToString().ToLower()
    return ($title.Contains("minecraft") -or $title.Contains("javaw") -or $title.Contains("az-launcher"))
}

function Get-IsMenuOpen {
     if (-not $script:onlyInGame) { return $false }
     $ci = New-Object WinApi+CURSORINFO
     $ci.cbSize = [System.Runtime.InteropServices.Marshal]::SizeOf($ci)
     if ([WinApi]::GetCursorInfo([ref]$ci)) { return ($ci.flags -eq 1) }
     return $false
 }

function Soda-Click-Delay($avgCPS) {
    if ($script:isBlatant) {
        $base = 1000 / $avgCPS
        $gauss = [InputSimulator]::NextGaussian(1.0, 0.15)
        $var = [Math]::Max(0.75, [Math]::Min(1.25, $gauss))
        $p = if ((Get-Random -Minimum 0 -Maximum 100) -gt 85) { Get-Random -Minimum 5 -Maximum 15 } else { 0 }
        return ($base * $var) + $p
    } else {
        $base = 1000 / $avgCPS
        $gauss = [InputSimulator]::NextGaussian(1.0, 0.08)
        $var = [Math]::Max(0.85, [Math]::Min(1.15, $gauss))
        $pause = 0
        $dice = Get-Random -Minimum 0 -Maximum 1000
        if ($dice -gt 990) { $pause = Get-Random -Minimum 50 -Maximum 150 }
        elseif ($dice -gt 950) { $pause = Get-Random -Minimum 10 -Maximum 30 }
        return ($base * $var) + $pause
    }
}

$script:nextL = 0; $script:nextR = 0; $script:lp = $false; $script:rp = $false; $script:hp = $false
$script:hue = 0
$script:leftMouseDown = $false
$script:rightMouseDown = $false
$mainTimer = New-Object System.Windows.Forms.Timer
$mainTimer.Interval = 1
$mainTimer.Add_Tick({
    $now = [DateTimeOffset]::Now.ToUnixTimeMilliseconds()
    $focused = Get-IsFocused
    $menu = Get-IsMenuOpen

    $script:hue = ($script:hue + 2) % 360
    $rainbowColor = [System.Drawing.Color]::FromArgb(
        [int](127 + 127 * [Math]::Sin($script:hue * [Math]::PI / 180)),
        [int](127 + 127 * [Math]::Sin(($script:hue + 120) * [Math]::PI / 180)),
        [int](127 + 127 * [Math]::Sin(($script:hue + 240) * [Math]::PI / 180))
    )
    $labelTitle.ForeColor = $rainbowColor
    $btnL.FlatAppearance.BorderColor = $rainbowColor
    $btnR.FlatAppearance.BorderColor = $rainbowColor
    $labelInfo.ForeColor = $rainbowColor

    if ($script:capturingLeftKey -or $script:capturingRightKey) {
        foreach ($key in $script:keyMap.Keys) {
            if ([GlobalHotkey]::IsKeyPressed($script:keyMap[$key])) {
                if ($script:capturingLeftKey) {
                    $script:leftBindKey = $script:keyMap[$key]
                    $btnL.Text = $key
                    $script:capturingLeftKey = $false
                } elseif ($script:capturingRightKey) {
                    $script:rightBindKey = $script:keyMap[$key]
                    $btnR.Text = $key
                    $script:capturingRightKey = $false
                }
                return
            }
        }
    }

    if ($script:leftBindKey -ne 0) {
        $isL = [GlobalHotkey]::IsKeyPressed($script:leftBindKey)
        if ($isL -and -not $script:lp) { $script:leftEnabled = -not $script:leftEnabled }
        $script:lp = $isL
    }
    if ($script:rightBindKey -ne 0) {
        $isR = [GlobalHotkey]::IsKeyPressed($script:rightBindKey)
        if ($isR -and -not $script:rp) { $script:rightEnabled = -not $script:rightEnabled }
        $script:rp = $isR
    }

    # F6 = hide/show
    $ish = [GlobalHotkey]::IsKeyPressed($script:hideKey)
    if ($ish -and -not $script:hp) {
        $script:isStealth = -not $script:isStealth
        if ($script:isStealth) { $form.Hide() } else { $form.Show() }
    }
    $script:hp = $ish

    $lS = if ($script:leftEnabled) { "ON" } else { "OFF" }
    $rS = if ($script:rightEnabled) { "ON" } else { "OFF" }
    $labelStatus.Text = "STATUS: L: $lS | R: $rS"
    $labelStatus.ForeColor = if ($script:leftEnabled -or $script:rightEnabled) { [System.Drawing.Color]::Lime } else { [System.Drawing.Color]::Gray }

    if ($focused -and -not $menu) {
        # LEFT MOUSE
        $leftPhysicalPressed = [GlobalHotkey]::IsKeyPressed(0x01)
        if ($script:leftEnabled -and $leftPhysicalPressed) {
            if (-not $script:leftMouseDown) {
                [InputSimulator]::LeftMouseDown()
                $script:leftMouseDown = $true
                $script:nextL = $now + (Soda-Click-Delay $script:leftCPS)
            }
            elseif ($now -ge $script:nextL) {
                [InputSimulator]::LeftMouseUp()
                Start-Sleep -Milliseconds 5
                [InputSimulator]::LeftMouseDown()
                if ($script:isBlockHit -and ($now -ge $script:lastBlockHit)) {
                    [InputSimulator]::BlockHit()
                    $script:lastBlockHit = $now + (Get-Random -Minimum 200 -Maximum 500)
                }
                $script:nextL = $now + (Soda-Click-Delay $script:leftCPS)
            }
        } else {
            if ($script:leftMouseDown) {
                [InputSimulator]::LeftMouseUp()
                $script:leftMouseDown = $false
            }
        }

        # RIGHT MOUSE
        $rightPhysicalPressed = [GlobalHotkey]::IsKeyPressed(0x02)
        if ($script:rightEnabled -and $rightPhysicalPressed) {
            if (-not $script:rightMouseDown) {
                [InputSimulator]::RightMouseDown()
                $script:rightMouseDown = $true
                $script:nextR = $now + (Soda-Click-Delay $script:rightCPS)
            }
            elseif ($now -ge $script:nextR) {
                [InputSimulator]::RightMouseUp()
                Start-Sleep -Milliseconds 5
                [InputSimulator]::RightMouseDown()
                $script:nextR = $now + (Soda-Click-Delay $script:rightCPS)
            }
        } else {
            if ($script:rightMouseDown) {
                [InputSimulator]::RightMouseUp()
                $script:rightMouseDown = $false
            }
        }
    }
})
$mainTimer.Start()

$form.KeyPreview = $true
$form.ShowDialog()
