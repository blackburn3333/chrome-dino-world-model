"""
Filename: dino.py
Author: Jayendra Matarage
Created on: 10/9/2026 7:29 AM
Description: Provides a Selenium-driven Chrome Dino environment with image observations and actions.
"""
import base64
import time
from typing import Any, Dict, Tuple
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import random
from PIL import Image
import io

class ChromeDinoEnv:
  """Expose Chrome's built-in Dino game through an environment-style API.

  The environment translates integer actions into browser keyboard events and
  converts the game's canvas into fixed-size grayscale NumPy arrays. Its
  ``reset`` and ``step`` methods follow the common reinforcement-learning
  environment pattern, although this class does not depend on Gym.
  """

  def __init__(self, frame_size: Tuple[int, int] = (64, 64)) -> None:
    """Start a Chrome session and locate the page element that receives input.

    Args:
      frame_size: Width and height used for returned grayscale observations.
    """
    self.frame_size = frame_size

    # Run Chrome offline so its built-in ``chrome://dino`` page is available,
    # and disable features that commonly fail in containers or CI environments.
    options = webdriver.ChromeOptions()
    options.add_argument("--mute-audio")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--offline")

    # Resolve a compatible ChromeDriver binary before creating the browser.
    service = Service(ChromeDriverManager().install())
    self.driver = webdriver.Chrome(service=service, options=options)

    # ChromeDriver can report an error for an internal chrome:// URL even when
    # the page loads correctly. Initialization continues and verifies the page
    # indirectly by locating its body below.
    try:
      self.driver.get("chrome://dino")
    except Exception:
      pass  # Expected exception when navigating offline

    # Allow the internal game page to initialize its canvas and Runner object.
    time.sleep(1)

    js_fill_screen = """
    var style = document.createElement('style');
    style.innerHTML = `
        body, html { margin: 0; padding: 0; overflow: hidden; width: 100%; height: 100%; }
        .runner-container { position: absolute !important; top: 0 !important; left: 0 !important; width: 100% !important; height: 100% !important; transform: none !important; }
        canvas.runner-canvas { width: 100% !important; height: 100% !important; }
    `;
    document.head.appendChild(style);
    """
    self.driver.execute_script(js_fill_screen)

    self.body = self.driver.find_element(By.TAG_NAME, "body")

  def reset(self) -> np.ndarray:
    """Restart a crashed game, start the runner, and return its first frame."""
    # The Runner instance is exposed in the page's JavaScript context. Restart
    # only after a collision; a fresh game does not need to be restarted.
    js_script = """
      try {
          if (typeof runnerInstance !== 'undefined' && runnerInstance.crashed) {
              runnerInstance.restart();
          }
      } catch (e) {}
      """
    try:
      self.driver.execute_script(js_script)
    except Exception:
      # A missing/unready Runner should not prevent the SPACE fallback below.
      pass

    # SPACE starts a new runner and is harmless if the game is already active.
    self.body.send_keys(Keys.SPACE)
    # Give the game time to advance before capturing the initial observation.
    time.sleep(0.5)

    return self._get_screen_frame()

  def moment(self, frame: int, frame_n: int, pre_action: int = None) -> int:
    """Select a weighted random action at a fixed frame interval.

    Args:
      frame: Current loop/frame index.
      frame_n: Number of frames for which an action should be retained.
      pre_action: Action selected at the previous decision point, or ``None``
        when selecting the initial action.

    Returns:
      The newly selected action on an interval boundary; otherwise the previous
      action so that input remains stable between decisions.
    """
    # These labels are used only to make the demonstration output readable.
    action_names = {
      0: "RUN",
      1: "JUMP",
      2: "CROUCH/DUCK",
      3: "FAST DROP"
    }

    if pre_action is None:
      # Always begin in the neutral RUN state before random actions are sampled.
      print(f"Step {frame}: Selected Action -> Initial: RUN")
      return 0

    # Choose a new action only at decision boundaries. On all other frames the
    # previous action is returned below.
    if frame % frame_n == 0:
      if pre_action == 1:
        # After a jump, favor neutral running while still allowing another jump
        # or a fast drop. Ducking is excluded from this airborne action set.
        actions_set = [0, 1, 3]
        weights = [0.50, 0.20, 0.30]
      else:
        # On the ground, select between running, jumping, and crouching.
        actions_set = [0, 1, 2]
        # Give CROUCH a distinct 30% probability.
        weights = [0.40, 0.30, 0.30]

      # ``random.choices`` returns a list even for one draw, hence index zero.
      current_action = random.choices(actions_set, weights=weights)[0]
      print(f"Step {frame}: Selected Action -> {action_names[current_action]}")
      return current_action

    # Maintain the current action between sampling intervals.
    return pre_action

  def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """Apply one action and return the resulting environment transition.

    Action mapping:
      0 = RUN (Release duck/jump keys)
      1 = JUMP (Spacebar / ArrowUp)
      2 = CROUCH / DUCK (Hold ArrowDown)
      3 = FAST DROP (Press ArrowDown while airborne)

    Returns:
      A tuple containing the processed frame, scalar reward, terminal flag, and
      an empty metadata dictionary reserved for future diagnostic information.
    """
    if action == 0:
        # Release ArrowDown so a prior crouch or fast-drop input does not stick.
        self.driver.execute_script(
            "document.dispatchEvent(new KeyboardEvent('keyup', {'key': 'ArrowDown', 'keyCode': 40, 'which': 40}));"
        )
    elif action == 1:
        # SPACE triggers the same jump behavior as ArrowUp in the Dino game.
        self.body.send_keys(Keys.SPACE)
    elif action in (2, 3):
        # The game interprets ArrowDown according to state: crouch while on the
        # ground, or descend faster while airborne.
        self.driver.execute_script(
            "document.dispatchEvent(new KeyboardEvent('keydown', {'key': 'ArrowDown', 'keyCode': 40, 'which': 40}));"
        )

    # Capture the observation after input, then query the terminal game state.
    frame = self._get_screen_frame()
    is_crashed = self._check_crashed()

    # Survival earns a small positive reward; collisions receive a penalty and
    # signal that the caller should reset the episode.
    reward = -1.0 if is_crashed else 0.1
    done = is_crashed

    return frame, reward, done, {}

  def _check_crashed(self) -> bool:
    """Return the Runner's collision flag, defaulting safely to ``False``."""
    # Querying JavaScript state is more reliable than inferring a collision from
    # pixels in the captured canvas.
    js_script = """
      try {
          if (typeof runnerInstance !== 'undefined') {
              return Boolean(runnerInstance.crashed);
          }
      } catch (e) {
          return false;
      }
      return false;
      """
    try:
      return bool(self.driver.execute_script(js_script))
    except Exception:
      # Treat transient script/driver failures as non-terminal observations.
      return False

  def _get_screen_frame(self) -> np.ndarray:
    """Captures active playfield (Dino + incoming obstacle track) and sharpens into a 64x64 frame."""
    try:
      container = self.driver.find_element(
        By.CSS_SELECTOR, "div.runner-container"
      )
      element_png = container.screenshot_as_png

      image = Image.open(io.BytesIO(element_png))
      img_np = np.array(image)

      # 1. Convert to Grayscale
      if len(img_np.shape) == 3:
        gray = cv2.cvtColor(
          img_np,
          cv2.COLOR_RGBA2GRAY
          if img_np.shape[2] == 4
          else cv2.COLOR_RGB2GRAY,
        )
      else:
        gray = img_np

      # 2. Invert dark mode so background is light (255) and game elements are dark (0)
      if gray[0, 0] < 128:
        gray = cv2.bitwise_not(gray)

      # 3. WIDER CROP: Capture Dino (left) + active obstacle area (right)
      # The canvas is roughly 600x150. Crop the first 350px horizontally.
      h, w = gray.shape[:2]
      crop_w = min(350, w)
      roi = gray[:, :crop_w]

      # 4. Sharpen image with thresholding to make sprites crisp
      _, thresh = cv2.threshold(roi, 210, 255, cv2.THRESH_BINARY)

      # 5. Downsample to target 64x64 resolution
      resized = cv2.resize(thresh, self.frame_size, interpolation=cv2.INTER_AREA)
      return resized

    except Exception:
      return np.zeros(self.frame_size, dtype=np.uint8)

  def close(self) -> None:
    """Close the Chrome window and terminate its WebDriver session."""
    self.driver.quit()

  def __enter__(self) -> "ChromeDinoEnv":
    """Return this environment when entering a ``with`` block."""
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Release browser resources when leaving a ``with`` block."""
    self.close()
