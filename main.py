"""
Flappy Bird — All Features Edition (All upgrades combined)
- 3D Parallax Neon City Background (simulated layers)
- Power-ups (magnet, slow, shield, double-jump)
- AI birds (simple competing birds)
- Dynamic Weather (rain, thunder, snow)
- Save highscores locally
- Neon UI + Menus
- Improved Face Control (mediapipe if available)
- On-screen touch buttons for mobile-like controls
- Adaptive difficulty (simple ML-like scaling)

NOTE: This is a large single-file Pygame project. Some optional features require
- mediapipe + opencv for face-control
- pygame 2.x

Run: python flappy_all_upgrades.py
"""

import pygame
import random
import sys
import math
import time
import threading
import json
import os

import numpy as np

# Optional imports
USE_MEDIAPIPE = True
try:
    import cv2
    import mediapipe as mp
except Exception:
    cv2 = None
    mp = None
    USE_MEDIAPIPE = False

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 720, 900
FPS = 60

GRAVITY = 0.38
JUMP_FORCE = -9.6
PIPE_SPEED_BASE = 3.5
PIPE_GAP_BASE = 220
PIPE_DISTANCE_BASE = 380

BIRD_RADIUS = 24
GROUND_HEIGHT = 96

BG_COLOR = (10, 12, 22)
NEON_COLOR = (60, 255, 200)

SAVE_FILE = 'flappy_highscores.json'

# ---------------- PYGAME INIT ----------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy — All Upgrades")
clock = pygame.time.Clock()

# fonts
font_big = pygame.font.SysFont("Segoe UI Black", 36)
font_mid = pygame.font.SysFont("Segoe UI", 28)
font_small = pygame.font.SysFont("Segoe UI", 18)

# sounds (synth)
pygame.mixer.init(frequency=44100, size=-16, channels=2)

def make_sound(freq=440.0, length_ms=120, volume=0.18):
    sr = 44100
    t = np.linspace(0, length_ms/1000.0, int(sr * (length_ms/1000.0)), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    win = np.hanning(len(wave))
    wave = wave * win
    wave_stereo = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound((wave_stereo * 32767 * volume).astype(np.int16))
    return sound

sound_flap = make_sound(720, 85, 0.15)
sound_score = make_sound(1400, 120, 0.18)
sound_hit = make_sound(120, 280, 0.26)
sound_power = make_sound(900, 140, 0.14)

# ---------------- UTIL ----------------

def clamp(v, a, b):
    return max(a, min(b, v))

# ---------------- BACKGROUND: 3D PARALLAX CITY ----------------
class CityLayer:
    def __init__(self, depth, color, building_min_w=40, building_max_w=120):
        self.depth = depth
        self.color = color
        self.buildings = []
        self.speed = 0.5 * depth
        self._populate(building_min_w, building_max_w)

    def _populate(self, min_w, max_w):
        x = 0
        while x < WIDTH + 200:
            w = random.randint(min_w, max_w)
            h = int((random.random()**1.5) * (HEIGHT//3) * self.depth) + 60
            y = HEIGHT - GROUND_HEIGHT - h
            glow = random.choice([0,1,0,0])
            self.buildings.append({'x':x,'w':w,'h':h,'y':y,'glow':glow})
            x += w + random.randint(6, 18)

    def update(self, speed_factor=1.0):
        for b in self.buildings:
            b['x'] -= self.speed * speed_factor
        # recycle
        if self.buildings and self.buildings[0]['x'] < -200:
            self.buildings.pop(0)
            # append new
            w = random.randint(40, 120)
            h = int((random.random()**1.5) * (HEIGHT//3) * self.depth) + 60
            y = HEIGHT - GROUND_HEIGHT - h
            x = self.buildings[-1]['x'] + self.buildings[-1]['w'] + random.randint(6, 18)
            self.buildings.append({'x':x,'w':w,'h':h,'y':y,'glow': random.choice([0,1,0])})

    def draw(self, surf):
        for b in self.buildings:
            rect = pygame.Rect(int(b['x']), b['y'], b['w'], b['h'])
            col = tuple(int(c * (0.12 + 0.6*self.depth)) for c in self.color)
            pygame.draw.rect(surf, col, rect)
            if b['glow']:
                # neon windows/dots
                for i in range(3):
                    wx = rect.x + 6 + i*20
                    wy = rect.y + 12
                    if wx < rect.right - 6:
                        pygame.draw.rect(surf, NEON_COLOR, (wx, wy, 8, 6))

city_layers = [CityLayer(0.35, (40,40,60)), CityLayer(0.6, (30,30,50)), CityLayer(1.2, (20,20,40))]

# ---------------- CLOUDS ----------------
class Cloud:
    def __init__(self, x,y, scale, speed):
        self.x=x; self.y=y; self.scale=scale; self.speed=speed
        self.w=int(140*scale); self.h=int(70*scale)
    def update(self):
        self.x -= self.speed
        if self.x < -self.w - 20:
            self.x = WIDTH + random.randint(0,200)
            self.y = random.randint(30, HEIGHT//2)
    def draw(self,surf):
        c=(230,230,240)
        pygame.draw.ellipse(surf,c,(self.x,self.y,self.w,self.h))
        pygame.draw.ellipse(surf,c,(self.x+int(self.w*0.2),self.y-int(self.h*0.3),int(self.w*0.7),int(self.h*0.8)))

clouds=[Cloud(random.randint(0,WIDTH),random.randint(10,HEIGHT//2),random.uniform(0.6,1.4),random.uniform(0.3,1.0)) for _ in range(6)]

# ---------------- BIRD SPRITE ----------------
def create_bird_frames(radius=24, frames=6):
    frames_list=[]
    for i in range(frames):
        surf=pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
        cx,cy=radius*2,radius*2
        pygame.draw.circle(surf,(255,235,80),(cx,cy),radius)
        # Flip beak to face right
        pygame.draw.polygon(surf,(255,140,20),[(cx+radius,cy),(cx+radius+10,cy-6),(cx+radius+10,cy+6)])
        angle = math.sin(i/frames*math.pi*2)*1.0
        wing_dy = int((radius*0.6)*math.sin(angle))
        # Flip wing to right side
        pygame.draw.ellipse(surf,(230,200,60),(cx-radius*0.8,cy-8+wing_dy,radius*1.6,radius*0.9))
        # Flip eye to right side
        pygame.draw.circle(surf,(20,20,20),(cx+6,cy-8),3)
        # Mirror the entire sprite horizontally
        surf = pygame.transform.flip(surf, True, False)
        frames_list.append(surf)
    return frames_list

BIRD_FRAMES = create_bird_frames(BIRD_RADIUS, frames=6)

# ---------------- PIPE (neon) ----------------
def draw_neon_pipe(surface, rect, glow_color=NEON_COLOR):
    rx,ry,rw,rh = rect
    glow = pygame.Surface((rw+40, rh+40), pygame.SRCALPHA)
    for i in range(10,0,-1):
        alpha = int(18 * (i/10))
        expand = int(i*2)
        pygame.draw.rect(glow, (*glow_color, alpha), pygame.Rect(20-expand,20-expand,rw+expand*2,rh+expand*2), border_radius=8)
    body = pygame.Surface((rw,rh), pygame.SRCALPHA)
    pygame.draw.rect(body, (*glow_color, 230), pygame.Rect(0,0,rw,rh), border_radius=6)
    inner=(12,12,12,210)
    pygame.draw.rect(body, inner, pygame.Rect(6,6,rw-12,rh-12), border_radius=5)
    surface.blit(glow,(rx-20,ry-20))
    surface.blit(body,(rx,ry))

# ---------------- PIPE CLASS ----------------
class Pipe:
    def __init__(self, x, width=96, top_min=90, gap=PIPE_GAP_BASE):
        self.x = x
        self.width = width
        self.gap = gap
        self.top_height = random.randint(top_min, HEIGHT - gap - 260)
        self.passed = False
    def update(self, speed):
        self.x -= speed
    def draw(self, surf):
        top_rect = (self.x, 0, self.width, self.top_height)
        bottom_rect = (self.x, self.top_height + self.gap, self.width, HEIGHT - (self.top_height + self.gap) - GROUND_HEIGHT)
        draw_neon_pipe(surf, top_rect)
        draw_neon_pipe(surf, bottom_rect)
    def collide(self, bx, by, r):
        top = (self.x, 0, self.width, self.top_height)
        bottom = (self.x, self.top_height + self.gap, self.width, HEIGHT - (self.top_height + self.gap) - GROUND_HEIGHT)
        return circle_rect_collision(bx, by, r, *top) or circle_rect_collision(bx, by, r, *bottom)

# ---------------- COLLISION UTIL ----------------
def circle_rect_collision(cx, cy, r, rx, ry, rw, rh):
    closest_x = clamp(cx, rx, rx + rw)
    closest_y = clamp(cy, ry, ry + rh)
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx*dx + dy*dy) <= (r*r)

# ---------------- POWER-UPS ----------------
class PowerUp:
    TYPES = ['magnet','slow','shield','double']
    def __init__(self, x):
        self.x = x
        self.y = random.randint(140, HEIGHT - GROUND_HEIGHT - 180)
        self.type = random.choice(PowerUp.TYPES)
        self.radius = 16
        self.col = {
            'magnet': (240,180,60), 'slow': (120,200,255), 'shield': (180,120,255), 'double': (120,255,140)
        }[self.type]
        self.active = True
    def update(self, speed):
        self.x -= speed
    def draw(self, surf):
        if not self.active: return
        pygame.draw.circle(surf, self.col, (int(self.x), int(self.y)), self.radius)
        txt = font_small.render(self.type[0].upper(), True, (10,10,10))
        surf.blit(txt, (self.x - txt.get_width()//2, self.y - txt.get_height()//2))
    def collide(self, bx, by, r):
        dx = bx - self.x; dy = by - self.y
        return (dx*dx + dy*dy) <= ((r+self.radius)**2)

# ---------------- AI BIRD (simple competitor) ----------------
class AIBird:
    def __init__(self, start_x, color=(200,120,90)):
        self.x = start_x
        self.y = random.randint(HEIGHT//3, HEIGHT - GROUND_HEIGHT - 120)
        self.vel = 0.0
        self.color = color
        self.radius = BIRD_RADIUS-6
        self.frame_idx = random.randint(0, len(BIRD_FRAMES)-1)
        self.frame_timer = 0
        self.score = 0
    def update(self, pipes, speed, difficulty):
        # naive AI: try to go towards center of next gap
        self.vel += GRAVITY * 0.9
        # find next pipe
        next_pipe = None
        for p in pipes:
            if p.x + p.width > self.x - 10:
                next_pipe = p
                break
        if next_pipe:
            target_y = next_pipe.top_height + next_pipe.gap/2
            # proportional control
            diff = target_y - self.y
            # more aggressive with difficulty
            if abs(diff) > 40 + difficulty*10:
                # flap if below target
                if diff < 0:
                    self.vel = JUMP_FORCE * (0.8 + difficulty*0.06)
        # update
        self.vel = clamp(self.vel, -999, 12 + difficulty*2)
        self.y += self.vel
        self.x -= speed * (0.9 + difficulty*0.05)
        # animation
        self.frame_timer += 1
        if self.frame_timer > 10:
            self.frame_idx = (self.frame_idx + 1) % len(BIRD_FRAMES)
            self.frame_timer = 0
    def draw(self, surf):
        f = BIRD_FRAMES[self.frame_idx]
        rot = pygame.transform.rotozoom(f, -self.vel*3, 0.9)
        rx, ry = rot.get_size()
        surf.blit(rot, (int(self.x - rx//2), int(self.y - ry//2)))

# ---------------- FACE CONTROL (improved) ----------------
FACE_PITCH_JUMP_THRESH = -7.0
FACE_SMOOTHING = 6
class FaceController:
    def __init__(self, cam=0):
        self.enabled = False
        self.cap = None
        self.face_mesh = None
        self.mp_face = None
        self.pitch_smooth = []
        self.avg_pitch = 0.0
        self.last_jump_time = 0
        if USE_MEDIAPIPE and mp and cv2:
            try:
                self.cap = cv2.VideoCapture(cam)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.mp_face = mp.solutions.face_mesh
                self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                       refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self.enabled = True
                self._stop = False
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            except Exception as e:
                print('Face init failed', e)
                self.enabled = False
        else:
            print('Mediapipe not available -> face disabled')
    def _run(self):
        while not getattr(self, '_stop', True):
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03); continue
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img)
            if not results.multi_face_landmarks:
                continue
            lm = results.multi_face_landmarks[0].landmark
            # better pitch estimate: average nose tip and forehead vs chin
            nose = lm[1]
            forehead = lm[10] if len(lm) > 10 else lm[10%len(lm)]
            chin = lm[152] if len(lm) > 152 else lm[-1]
            eyeL = lm[33]; eyeR = lm[263]
            eye_mid_x = (eyeL.x + eyeR.x)/2; eye_mid_y = (eyeL.y + eyeR.y)/2
            dx = (nose.x + forehead.x)/2 - eye_mid_x
            dy = (nose.y + forehead.y)/2 - eye_mid_y
            pitch = math.degrees(math.atan2(dy, dx))
            self.pitch_smooth.append(pitch)
            if len(self.pitch_smooth) > FACE_SMOOTHING:
                self.pitch_smooth.pop(0)
            self.avg_pitch = sum(self.pitch_smooth)/len(self.pitch_smooth)
            time.sleep(0.02)
    def wants_jump(self):
        if not self.enabled: return False
        now = time.time()
        if self.avg_pitch < FACE_PITCH_JUMP_THRESH and (now - self.last_jump_time) > 0.45:
            self.last_jump_time = now
            return True
        return False
    def close(self):
        try:
            self._stop=True
            if self.cap: self.cap.release()
            if self.face_mesh: self.face_mesh.close()
        except Exception:
            pass

# ---------------- SAVE HIGHSCORES ----------------
def load_scores():
    if not os.path.exists(SAVE_FILE):
        return {'high':0,'history':[]}
    try:
        with open(SAVE_FILE,'r') as f:
            return json.load(f)
    except Exception:
        return {'high':0,'history':[]}

def save_scores(data):
    try:
        with open(SAVE_FILE,'w') as f:
            json.dump(data,f)
    except Exception:
        pass

# ---------------- WEATHER SYSTEM ----------------
class Weather:
    MODES = ['none','rain','snow','storm']
    def __init__(self):
        self.mode='none'
        self.drop_pixels = []
        self.timer=0
    def set(self,m):
        self.mode=m
        self.drop_pixels=[]
    def update(self):
        if self.mode=='rain':
            # spawn drops
            if random.random() < 0.6:
                self.drop_pixels.append([random.randint(0,WIDTH), -10, random.randint(400,700)])
            for d in list(self.drop_pixels):
                d[1]+=12
                if d[1] > HEIGHT: self.drop_pixels.remove(d)
        elif self.mode=='snow':
            if random.random() < 0.4:
                self.drop_pixels.append([random.randint(0,WIDTH), -10, random.uniform(1,3)])
            for d in list(self.drop_pixels):
                d[1]+=d[2]
                d[0]+=math.sin(time.time()+d[1])*0.3
                if d[1] > HEIGHT: self.drop_pixels.remove(d)
        elif self.mode=='storm':
            # mix of rain + occasional flash
            if random.random() < 0.8:
                self.drop_pixels.append([random.randint(0,WIDTH), -10, random.randint(500,800)])
            for d in list(self.drop_pixels):
                d[1]+=16
                if d[1] > HEIGHT: self.drop_pixels.remove(d)
            # thunder flash occasionally
            if random.random() < 0.02:
                self.timer = 6
        if self.timer>0: self.timer-=1
    def draw(self,surf):
        if self.mode in ['rain','storm']:
            for d in self.drop_pixels:
                pygame.draw.line(surf, (150,180,200), (d[0], d[1]), (d[0], d[1]+6), 1)
        if self.mode=='snow':
            for d in self.drop_pixels:
                pygame.draw.circle(surf, (240,240,250), (int(d[0]), int(d[1])), 3)
        if self.timer>0:
            overlay = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
            overlay.fill((255,255,255, 40 + self.timer*6))
            surf.blit(overlay, (0,0))

weather = Weather()

# ---------------- ADAPTIVE DIFFICULTY ----------------
class AdaptiveDifficulty:
    def __init__(self):
        self.history = []  # recent scores
    def update(self, score):
        self.history.append(score)
        if len(self.history)>10: self.history.pop(0)
    def difficulty(self):
        # naive: slope of scores
        if len(self.history) < 2: return 0.0
        dif = self.history[-1] - self.history[0]
        d = clamp(dif / 10.0, 0.0, 3.0)
        return d

adaptive = AdaptiveDifficulty()

# ---------------- MAIN GAME ----------------
def draw_background(surface):
    surface.fill(BG_COLOR)
    # clouds
    for c in clouds: c.draw(surface)
    # city layers
    for layer in city_layers:
        layer.draw(surface)

def draw_ground(surface):
    pygame.draw.rect(surface, (18,16,20), (0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT))
    # neon stripe
    for i in range(0, WIDTH, 40):
        pygame.draw.rect(surface, (10,200,170), (i, HEIGHT - GROUND_HEIGHT + 8, 24, 6))

def draw_score(surface, score):
    txt = font_mid.render(str(score), True, (200, 255, 240))
    surface.blit(txt, (WIDTH//2 - txt.get_width()//2, 24))

# on-screen touch flop button
BTN_RECT = pygame.Rect(WIDTH-120, HEIGHT-120, 96, 96)

def main():
    bird_x = 140
    bird_y = HEIGHT//2
    bird_vel = 0.0
    bird_frame_idx = 0
    bird_anim_timer = 0

    pipes = [Pipe(WIDTH + 240, gap=PIPE_GAP_BASE)]
    powerups = []
    ai_birds = [AIBird(WIDTH - 150, color=(220,120,100)), AIBird(WIDTH - 300, color=(150,220,140))]

    score = 0
    running = True
    state = 'MENU'

    face_ctrl = FaceController() if USE_MEDIAPIPE else None
    # Use a list to make it mutable in the event loop
    face_enabled = [False]
    if face_ctrl and face_ctrl.enabled:
        face_enabled[0] = True

    spawn_timer = 0
    last_time = pygame.time.get_ticks()

    speed = PIPE_SPEED_BASE
    gap = PIPE_GAP_BASE
    distance = PIPE_DISTANCE_BASE

    highdata = load_scores()
    highscore = highdata.get('high', 0)

    weather_modes = ['none','rain','snow','storm']
    weather.set('none')

    difficulty_agent = adaptive

    magnet_active = False
    magnet_until = 0
    slow_active = False
    slow_until = 0
    shield_active = False
    shield_until = 0
    double_active = False
    double_until = 0

    while running:
        dt_ms = clock.tick(FPS)
        t = pygame.time.get_ticks() / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_f:
                    if face_ctrl and face_ctrl.enabled:
                        face_enabled[0] = not face_enabled[0]
                        print(f"Face control: {'ON' if face_enabled[0] else 'OFF'}")
                if event.key == pygame.K_SPACE:
                    if state == 'MENU':
                        state = 'PLAYING'
                        bird_vel = JUMP_FORCE * 0.6
                        sound_flap.play()
                    elif state == 'PLAYING':
                        bird_vel = JUMP_FORCE
                        sound_flap.play()
                    elif state == 'GAMEOVER':
                        pipes = [Pipe(WIDTH + 240, gap=gap)]
                        powerups = []
                        bird_y = HEIGHT//2
                        bird_vel = 0.0
                        score = 0
                        state = 'MENU'
                if event.key == pygame.K_m:
                    # cycle weather for demo
                    idx = weather_modes.index(weather.mode)
                    weather.set(weather_modes[(idx+1)%len(weather_modes)])
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx,my = event.pos
                if BTN_RECT.collidepoint(mx,my):
                    # on-screen flap
                    if state == 'MENU':
                        state = 'PLAYING'; bird_vel = JUMP_FORCE * 0.6; sound_flap.play()
                    elif state == 'PLAYING':
                        bird_vel = JUMP_FORCE; sound_flap.play()
                    elif state == 'GAMEOVER':
                        pipes = [Pipe(WIDTH + 240, gap=gap)]; powerups=[]; bird_y = HEIGHT//2; bird_vel=0; score=0; state='MENU'
                else:
                    # normal mouse click
                    if state == 'MENU':
                        state = 'PLAYING'; bird_vel = JUMP_FORCE * 0.6; sound_flap.play()
                    elif state == 'PLAYING':
                        bird_vel = JUMP_FORCE; sound_flap.play()
                    elif state == 'GAMEOVER':
                        pipes = [Pipe(WIDTH + 240, gap=gap)]; powerups=[]; bird_y = HEIGHT//2; bird_vel=0; score=0; state='MENU'

        # face control
        if state == 'PLAYING' and face_enabled[0] and face_ctrl:
            if face_ctrl.wants_jump():
                bird_vel = JUMP_FORCE; sound_flap.play()

        # update city and clouds
        # difficulty modifies background speed
        diff = difficulty_agent.difficulty()
        for layer in city_layers:
            layer.update(1.0 + diff*0.08)
        for c in clouds: c.update()

        # weather update
        weather.update()

        # update gameplay when playing
        if state == 'PLAYING':
            # adjust dynamic speed/gap
            speed = PIPE_SPEED_BASE + diff*0.6
            gap = int(PIPE_GAP_BASE - diff*10)
            distance = int(PIPE_DISTANCE_BASE - diff*12)

            bird_vel += GRAVITY
            bird_vel = clamp(bird_vel, -999, 16)
            bird_y += bird_vel

            # spawn pipes and powerups
            if pipes[-1].x < WIDTH - distance:
                pipes.append(Pipe(WIDTH + 40, gap=gap))
                # occasionally spawn powerup
                if random.random() < 0.25:
                    powerups.append(PowerUp(WIDTH + 40 + random.randint(40,120)))

            for p in pipes:
                p.update(speed)
            for pu in list(powerups):
                pu.update(speed)
                if pu.x < -50: powerups.remove(pu)

            # remove off-screen pipes
            pipes = [p for p in pipes if p.x > -200]

            # AI birds update
            for ai in ai_birds:
                ai.update(pipes, speed, diff)

            # scoring and collisions
            collided = False
            cx = bird_x; cy = bird_y
            for p in pipes:
                if not p.passed and (p.x + p.width) < cx:
                    p.passed = True
                    # score increments; double powerup doubles score
                    inc = 2 if double_active and time.time() < double_until else 1
                    score += inc
                    sound_score.play()
                if p.collide(cx, cy, BIRD_RADIUS):
                    collided = True

            # pick powerups
            for pu in list(powerups):
                if pu.collide(cx, cy, BIRD_RADIUS):
                    sound_power.play()
                    pu.active = False
                    try: powerups.remove(pu)
                    except: pass
                    now = time.time()
                    if pu.type == 'magnet':
                        magnet_active = True; magnet_until = now + 6
                    elif pu.type == 'slow':
                        slow_active = True; slow_until = now + 5
                    elif pu.type == 'shield':
                        shield_active = True; shield_until = now + 5
                    elif pu.type == 'double':
                        double_active = True; double_until = now + 6

            # magnet effect: pull score or collect nearby powerups
            if magnet_active and time.time() < magnet_until:
                for pu in list(powerups):
                    # move powerups slightly towards bird
                    pu.x += (bird_x - pu.x) * 0.08
            else:
                magnet_active = False

            # slow effect reduces speed
            if slow_active and time.time() < slow_until:
                speed = max(1.6, speed * 0.6)
            else:
                slow_active = False

            # ground/ceiling
            if cy - BIRD_RADIUS <= 0 or cy + BIRD_RADIUS >= HEIGHT - GROUND_HEIGHT:
                collided = True

            if collided:
                if shield_active and time.time() < shield_until:
                    shield_active = False
                    sound_hit.play()
                else:
                    sound_hit.play()
                    state = 'GAMEOVER'
                    # save score
                    if score > highscore:
                        highscore = score
                        highdata['high'] = highscore
                    highdata.setdefault('history',[]).append({'score':score,'time':int(time.time())})
                    save_scores(highdata)

            # adaptive difficulty update
            difficulty_agent.update(score)

        # DRAW
        draw_background(screen)
        # subtle grid
        for i in range(0, WIDTH, 140):
            pygame.draw.line(screen, (8,8,10), (i,0), (i, HEIGHT), 1)

        for p in pipes: p.draw(screen)
        for pu in powerups: pu.draw(screen)
        for ai in ai_birds: ai.draw(screen)

        # bird anim
        bird_anim_timer += dt_ms
        if bird_anim_timer > 90:
            bird_frame_idx = (bird_frame_idx + 1) % len(BIRD_FRAMES)
            bird_anim_timer = 0
        frame = BIRD_FRAMES[bird_frame_idx]
        rot = pygame.transform.rotozoom(frame, -bird_vel*3, 1.0)
        rx, ry = rot.get_size()
        screen.blit(rot, (bird_x - rx//2, int(bird_y) - ry//2))

        # shield visual
        if shield_active and time.time() < shield_until:
            pygame.draw.circle(screen, (120,220,255,80), (int(bird_x), int(bird_y)), BIRD_RADIUS+12, 2)

        draw_ground(screen)

        # HUD
        if state == 'MENU':
            title = font_big.render('FLAPPY NEON — ALL UPGRADES', True, (220,255,240))
            subtitle = font_mid.render('Space/Click or On-screen button to start', True, (200,200,200))
            hint = font_small.render("Press F to toggle face-control (if available) • M to cycle weather", True, (160,160,160))
            screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 160))
            screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, HEIGHT//2 - 60))
            screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT//2))
            hs = font_small.render(f'Highscore: {highscore}', True, (180,220,200))
            screen.blit(hs, (WIDTH//2 - hs.get_width()//2, HEIGHT//2 + 40))
        elif state == 'PLAYING':
            draw_score(screen, score)
        elif state == 'GAMEOVER':
            over = font_big.render('GAME OVER', True, (255,120,120))
            sub = font_mid.render(f'Score: {score}  High: {highscore}', True, (220,220,220))
            hint = font_small.render('Click/Space to restart • F to toggle face-control', True, (180,180,180))
            screen.blit(over, (WIDTH//2 - over.get_width()//2, HEIGHT//2 - 80))
            screen.blit(sub, (WIDTH//2 - sub.get_width()//2, HEIGHT//2))
            screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT//2 + 60))

        # face indicator
        if face_ctrl and face_ctrl.enabled:
            ico = font_small.render(('🤖 Face: ON' if face_enabled[0] else '🤖 Face: OFF'), True, (120,220,180))
            screen.blit(ico, (12, 12))

        # powerup status
        psts = []
        if magnet_active: psts.append('MAGNET')
        if slow_active: psts.append('SLOW')
        if shield_active: psts.append('SHIELD')
        if double_active: psts.append('DOUBLE')
        if psts:
            pstxt = font_small.render(' | '.join(psts), True, (220,240,220))
            screen.blit(pstxt, (12, 40))

        # on-screen button
        pygame.draw.rect(screen, (20,20,30), BTN_RECT, border_radius=14)
        pygame.draw.circle(screen, (80,200,160), (BTN_RECT.centerx, BTN_RECT.centery), 36)
        arrow = font_mid.render('FLAP', True, (8,8,8))
        screen.blit(arrow, (BTN_RECT.centerx - arrow.get_width()//2, BTN_RECT.centery - arrow.get_height()//2))

        # weather overlay
        weather.draw(screen)

        pygame.display.flip()

    # cleanup
    if face_ctrl: face_ctrl.close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
