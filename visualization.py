"""
This file is used for visualization. No big changes should be required to this file.
The only function that can be updated is map_running. You may add additional info to be printed such as a/c in the radar range yourself.
Visualization code in this file is based on graduation work by K.Fines and T. Noortman.
"""

import numpy as np
import pygame as pg
import os
import ctypes
import time as timer
import sys

# %% Initialize values
# User inputs (may be changed)
disp_agent_id = True  # display aircraft id number
disp_waypoint_id = True  # display waypoint id number
disp_radar_aircraft = False  # display which other aircraft an aircaft sees
disp_time = True  # display current time
disp_waypoint = True  # display dot of the waypoint
disp_vehicles = True

# Lines below are used in visualization, dont change them (line 26-46)
piclist = []  # list to store all pics
rectlist = []  # list to store the rectangular areas of the surface
tug_piclist = []  # list to store all tug pics
tug_rectlist = []  # list to store the rectangular areas of the tug surface
tug_ac_piclist = []  # list to store the combined tug+aircraft pics
tug_ac_rectlist = []  # list to store the rectangular areas

squared_display = True  # create squared display
boundary_margin = 0.01  # add boundary margin to be sure that points are within boundary
screen_percentage = 0.94  # percentage of pixels used in the critical axis
horizontal_sep = 0.01  # percentage of pixels used in between dual screen

red = (255, 0, 0)  # define colors
lightgreen = (0, 255, 147)
green = (0, 255, 0)
darkgreen = (0, 135, 0)
yellow = (255, 251, 45)
orange = (244, 149, 66)
pink = (255, 192, 203)
purple = (255, 0, 255)

blue = (0, 0, 255)
black = (10, 10, 10)
white = (255, 255, 255)
lightblue = (173, 216, 230)

# %% Functions


# function to convert x-coordinate location to pixel in screen
def c2m_x(x_coord, min_x, reso_x, x_range, shift=0):
    return int((float(x_coord - min_x) / x_range) * reso_x + shift)


# function to convert y-coordinate location to pixel in screen
def c2m_y(y_coord, max_y, reso_y, y_range, shift=0):
    return int(
        (
            (
                float(
                    # -1 is used since pixels are measured from top of screen (y = 0 at top)
                    y_coord
                    - max_y
                )
                / y_range
            )
            * reso_y
        )
        * -1
        - shift
    )


def plot_aircraft(scr, reso, deg, x, y, x0, y0, x_range, y_range, x_shift=0, y_shift=0):
    # convert x-coordinates to map coordinates
    plane_map_x = c2m_x(x, x0, reso[0], x_range, x_shift)
    # convert y-coordinates to map coordinates
    plane_map_y = c2m_y(y, y0, reso[1], y_range, y_shift)
    rectlist[deg].centerx = plane_map_x  # set x-location of the aircraft image
    rectlist[deg].centery = plane_map_y  # set y-location of the aircraft image
    # blit the aircraft image to the screen
    scr.blit(piclist[deg], rectlist[deg])


def plot_tug(scr, reso, deg, x, y, x0, y0, x_range, y_range, x_shift=0, y_shift=0):
    # convert x-coordinates to map coordinates
    tug_map_x = c2m_x(x, x0, reso[0], x_range, x_shift)
    # convert y-coordinates to map coordinates
    tug_map_y = c2m_y(y, y0, reso[1], y_range, y_shift)

    tug_rectlist[deg].centerx = tug_map_x  # set x-location of the tug image
    tug_rectlist[deg].centery = tug_map_y  # set y-location of the tug image
    # blit the tug image to the screen
    scr.blit(tug_piclist[deg], tug_rectlist[deg])


def plot_tug_aircraft(scr, reso, deg, x, y, x0, y0, x_range, y_range, x_shift=0, y_shift=0):
    # convert x-coordinates to map coordinates
    tug_ac_map_x = c2m_x(x, x0, reso[0], x_range, x_shift)
    # convert y-coordinates to map coordinates
    tug_ac_map_y = c2m_y(y, y0, reso[1], y_range, y_shift)

    tug_ac_rectlist[deg].centerx = tug_ac_map_x  # set x-location
    tug_ac_rectlist[deg].centery = tug_ac_map_y  # set y-location
    # blit the combined image to the screen
    scr.blit(tug_ac_piclist[deg], tug_ac_rectlist[deg])


def plot_line(scr, color_code, reso, radius, coord_1, coord_2, x0, y0, x_range, y_range, x_shift=0, y_shift=0):
    wp_map_x_1 = c2m_x(coord_1[0], x0, reso[0], x_range, x_shift)  # get x-pixel of source
    wp_map_y_1 = c2m_y(coord_1[1], y0, reso[1], y_range, y_shift)  # get y-pixel of source
    wp_map_x_2 = c2m_x(coord_2[0], x0, reso[0], x_range, x_shift)  # get x-pixel of target
    wp_map_y_2 = c2m_y(coord_2[1], y0, reso[1], y_range, y_shift)  # get y-pixel of target
    pg.draw.line(
        scr, color_code, (wp_map_x_1, wp_map_y_1), (wp_map_x_2, wp_map_y_2), radius
    )  # draw line between source and target


def plot_circle(scr, color_code, reso, radius, coord_1, x0, y0, x_range, y_range, x_shift=0, y_shift=0):
    wp_map_x = c2m_x(coord_1[0], x0, reso[0], x_range, x_shift)  # get x-pixel of circle center
    wp_map_y = c2m_y(coord_1[1], y0, reso[1], y_range, y_shift)  # get y-pixel of circle center
    # draw a dot on the screen at found coodrinates
    pg.draw.circle(scr, color_code, (wp_map_x, wp_map_y), radius)


def plot_text(scr, text_str, color_code, fontsize, reso, x, y, x0, y0, x_range, y_range, x_shift=0, y_shift=0):
    font = pg.font.Font(None, fontsize)  # set the font + font size
    text = font.render(
        text_str, 1, color_code
    )  # draw text on surface, using found density string, 1 = smooth edges, 10-10-10 is black color
    textpos = text.get_rect()  # get rectangle of text string
    # convert x-coordinates to map coordinates
    wp_map_x = c2m_x(x, x0, reso[0], x_range, x_shift)
    # convert y-coordinates to map coordinates
    wp_map_y = c2m_y(y, y0, reso[1], y_range, y_shift)
    textpos.centerx = wp_map_x  # determine x-location of text string
    textpos.centery = wp_map_y  # determine y-location of text string
    scr.blit(text, textpos)


def map_initialization(nodes_dict, edges_dict):  # function to initialise mapf
    # print(edges_dict)

    map_properties = dict()  # create dict to return all properties
    # get info about the screen range properties
    map_get_range(nodes_dict, map_properties)

    # line of code required to get correct resolution of screen
    if sys.platform.startswith("win"):
        # line of code required to get correct resolution of screen on Windows
        ctypes.windll.user32.SetProcessDPIAware()
        true_res = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
    else:
        # Linux or MacOS
        pg.display.init()
        display_info = pg.display.Info()
        true_res = (display_info.current_w, display_info.current_h)

    reso_ratio = map_properties["x_range"] / float(map_properties["y_range"])

    # calculate limiting axis: horizontal vs vertical
    x_pixels = int(screen_percentage * min(true_res[0], true_res[1] * reso_ratio))
    y_pixels = int(x_pixels / reso_ratio)  # get corresponding y_pixels
    outer_reso = (x_pixels, y_pixels)  # set resolution full interface screen
    # set resolution simulation screen within full interface
    inner_reso = (x_pixels, y_pixels)

    os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (
        int(0.5 * true_res[0] - 0.5 * outer_reso[0]),
        int(
            # make sure maps comes in middle screen
            0.5 * true_res[1]
            - 0.5 * outer_reso[1]
        ),
    )

    pg.init()  # initialise pygame
    scr = pg.display.set_mode(outer_reso)
    scrrect = scr.get_rect()  # get rectangular area of the surface
    scr.fill(white)  # set background color
    # get the aircraft image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plane_pic = pg.image.load(os.path.join(script_dir, "blue-plane-hi.bmp"))
    # remove white background to make transparent
    plane_pic.set_colorkey(pg.Color(255, 255, 255))

    for i in range(0, 360):  # transform aircraft image in every possible direction
        # 1/14 is used for scaling the aircraft image
        piclist.append(pg.transform.rotozoom(plane_pic, i, (1.0 / 14.0)))
        # get rectangular surface of the pic
        rectlist.append(piclist[i].get_rect())

    tug_pic = pg.image.load(os.path.join(script_dir, "tug.bmp"))
    tug_pic.set_colorkey(pg.Color(255, 255, 255))  # remove white background

    for i in range(0, 360):
        tug_piclist.append(pg.transform.rotozoom(tug_pic, i, (1.0 / 20.0)))
        tug_rectlist.append(tug_piclist[i].get_rect())

    # --- NEW: Load the combined Tug + Aircraft image ---
    tug_ac_pic = pg.image.load(os.path.join(script_dir, "tug_aircraft.bmp"))
    tug_ac_pic.set_colorkey(pg.Color(255, 255, 255))  # remove white background

    for i in range(0, 360):
        # 1/40.0 scaling to match the aircraft size
        tug_ac_piclist.append(pg.transform.rotozoom(tug_ac_pic, i, (1.0 / 20.0)))
        tug_ac_rectlist.append(tug_ac_piclist[i].get_rect())

    # store created information (resolution)
    map_properties["outer_reso"] = outer_reso
    map_properties["inner_reso"] = inner_reso  # resolution airport layout
    map_properties["scr"] = scr  # background
    map_properties["scrrect"] = scrrect  # surface of background
    map_properties["piclist"] = piclist  # aircraft data
    map_properties["rectlist"] = rectlist  # aircraft surface
    map_properties["tug_piclist"] = tug_piclist  # tug data
    map_properties["tug_ac_piclist"] = tug_ac_piclist
    map_properties["tug_ac_rectlist"] = tug_ac_rectlist

    # create the background layout
    map_get_background(map_properties, nodes_dict, edges_dict)

    return map_properties  # return all information


def map_get_range(nodes_dict, map_properties):

    min_x = np.inf
    max_x = 0
    min_y = np.inf
    max_y = 0
    for node in nodes_dict:
        if nodes_dict[node]["x_pos"] < min_x:
            min_x = nodes_dict[node]["x_pos"] - 3
        if nodes_dict[node]["x_pos"] > max_x:
            max_x = nodes_dict[node]["x_pos"] + 4
        if nodes_dict[node]["y_pos"] < min_y:
            min_y = nodes_dict[node]["y_pos"] - 2
        if nodes_dict[node]["y_pos"] > max_y:
            max_y = nodes_dict[node]["y_pos"] + 2

    x_range = max_x - min_x  # get difference between max and min x-coordinate
    y_range = max_y - min_y  # get difference between max and min y-coordinate

    if squared_display:  # in case a squared display has to be created, x + y have to be adapted
        if x_range > y_range:
            min_x -= int(boundary_margin * x_range)
            max_x += int(boundary_margin * x_range)
            diff_range = (max_x - min_x) - y_range
            min_y -= int(0.5 * diff_range)
            max_y += int(0.5 * diff_range)

        elif y_range > x_range:
            min_y -= int(boundary_margin * y_range)
            max_y += int(boundary_margin * y_range)
            diff_range = (max_y - min_y) - x_range
            min_x -= int(0.5 * diff_range)
            max_x += int(0.5 * diff_range)
        else:
            min_x -= int(boundary_margin * x_range)
            max_x += int(boundary_margin * x_range)
            min_y -= int(boundary_margin * y_range)
            max_y += int(boundary_margin * y_range)
    else:
        min_x -= int(boundary_margin * x_range)
        max_x += int(boundary_margin * x_range)
        min_y -= int(boundary_margin * y_range)
        max_y += int(boundary_margin * y_range)

    x_range = max_x - min_x  # get difference between max and min x-coordinate
    y_range = max_y - min_y  # get difference between max and min y-coordinate

    map_properties["min_x"] = min_x  # store all parameters
    map_properties["max_x"] = max_x
    map_properties["x_range"] = x_range
    map_properties["min_y"] = min_y
    map_properties["max_y"] = max_y
    map_properties["y_range"] = y_range


def map_get_background(map_properties, nodes_dict, edges_dict):
    reso = map_properties["inner_reso"]  # get resolution
    scr = map_properties["scr"]  # get screen

    min_x = map_properties["min_x"]  # get x0
    x_range = map_properties["x_range"]  # get horizontal range
    max_y = map_properties["max_y"]  # get y0 (measured from above)
    y_range = map_properties["y_range"]  # get vertical range

    map_get_layout(scr, nodes_dict, edges_dict, min_x, max_y, reso, x_range, y_range, 0, 0)

    # transform image to string (faster reading during simulation
    background = pg.image.tostring(scr, "RGB")
    map_properties["background"] = background  # store background


def map_get_layout(scr, nodes_dict, edges_dict, min_x, max_y, reso, x_range, y_range, scr_x_shift, scr_y_shift):
    # Print edges on map
    edges_created = set()
    for edge in edges_dict:
        if edge not in edges_created:
            source = edges_dict[edge]["start_end_pos"][0]
            target = edges_dict[edge]["start_end_pos"][1]
            plot_line(scr, black, reso, 1, source, target, min_x, max_y, x_range, y_range, scr_x_shift, scr_y_shift)
            edges_created.add((edges_dict[edge]["to"], edges_dict[edge]["from"]))
            edges_created.add((edges_dict[edge]["from"], edges_dict[edge]["to"]))
        else:
            pass

    # Print runways on map
    runway_a = [(3, 3), (3, 6)]
    runway_d = [(1, 4), (1, 7)]
    plot_line(
        scr, darkgreen, reso, 5, runway_a[0], runway_a[1], min_x, max_y, x_range, y_range, scr_x_shift, scr_y_shift
    )
    plot_line(
        scr, darkgreen, reso, 5, runway_d[0], runway_d[1], min_x, max_y, x_range, y_range, scr_x_shift, scr_y_shift
    )

    # --- NEW: Add labels for runways ---
    plot_text(
        scr,
        "Arrival",
        darkgreen,
        22,
        reso,
        runway_a[0][0] + 0.4,
        (runway_a[0][1] + runway_a[1][1]) / 2,
        min_x,
        max_y,
        x_range,
        y_range,
        scr_x_shift,
        scr_y_shift,
    )

    plot_text(
        scr,
        "Departure",
        darkgreen,
        22,
        reso,
        runway_d[0][0] - 0.8,
        (runway_d[0][1] + runway_d[1][1]) / 2,
        min_x,
        max_y,
        x_range,
        y_range,
        scr_x_shift,
        scr_y_shift,
    )

    # Print waypoints and ids on map
    for node in nodes_dict:
        wp_coordinate = [nodes_dict[node]["x_pos"], nodes_dict[node]["y_pos"]]
        color = blue
        plot_circle(scr, color, reso, 4, wp_coordinate, min_x, max_y, x_range, y_range)
        thisString = str(nodes_dict[node]["id"])  # create string with node ID
        plot_text(
            scr, thisString, black, 14, reso, wp_coordinate[0], wp_coordinate[1], min_x, max_y, x_range, y_range, 10, 10
        )


# %% Update map during running


def map_running(map_properties, current_states, t):  # function to update the map
    """
    Function updates Pygame map based on the map_properties, current state of the vehicles and the time.
    Collisions are detected if two tugs are at the same xy_position.
    If escape key is pressed, pygame closes.
    If "p" key is pressed, pygame pauses. If enter is pressed pygame continues.
    INPUT:
        - map_properties = dict with properties as created in map_intialization.
        - current_states = dict with id, heading and xy_pos of all active vehicles.
        - t = time.
    RETURNS:
        - Function updates pygame.
        - escape_pressed = boolean (True/False) = Used to end simulation loop if escape is pressed.
    """

    reso = map_properties["inner_reso"]  # get resolution
    scr = map_properties["scr"]  # get screen
    scrrect = map_properties["scrrect"]  # get screen surface

    min_x = map_properties["min_x"]  # get x0
    x_range = map_properties["x_range"]  # get horizontal range
    max_y = map_properties["max_y"]  # get y0 (measured from above)
    y_range = map_properties["y_range"]  # get vertical range
    background = map_properties["background"]  # get layout background

    # transform background from string to image
    layout = pg.image.fromstring(background, scrrect.size, "RGB")
    scr.blit(layout, scrrect)  # print layout on screen

    # draw vehicles and ids
    if disp_vehicles:
        for agent in current_states.keys():
            # Safely get heading, defaulting to 0 if it doesn't exist yet
            heading = int(current_states[agent].get("heading", 0))
            x_pos = current_states[agent]["xy_pos"][0]
            y_pos = current_states[agent]["xy_pos"][1]

            if current_states[agent].get("type") == "tug":
                plot_tug(scr, reso, heading, x_pos, y_pos, min_x, max_y, x_range, y_range)
            elif current_states[agent].get("type") == "tug_towing":
                plot_tug_aircraft(scr, reso, heading, x_pos, y_pos, min_x, max_y, x_range, y_range)
            else:
                plot_aircraft(scr, reso, heading, x_pos, y_pos, min_x, max_y, x_range, y_range)

    if disp_agent_id:  # if the aircraft/tug id has to be displayed
        for agent in current_states.keys():
            # Check the type to assign the right prefix and color
            if current_states[agent].get("type") == "tug_towing":
                # Show both IDs when coupled
                tug_id = current_states[agent].get("id")
                ac_id = current_states[agent].get("ac_id", "?")
                id_string = f"TUG: {tug_id} | AC: {ac_id}"
                col = blue

                # New Battery Line
                is_charging = current_states[agent].get("is_charging", False)
                # We use int() to keep the display clean, or :.1f if you want one decimal place
                battery_val = current_states[agent].get("battery", 999)
                # Update label to show (+) and use Green when charging
                if is_charging:
                    bat_string = f"BAT: {int(battery_val)}% (+)"
                    bat_col = lightgreen  # Bright green for charging
                else:
                    bat_string = f"BAT: {int(battery_val)}%"
                    # color based on battery health
                    # Green (>30), Yellow (10-30), or Red (<10)
                    bat_col = green if battery_val > 30 else yellow if battery_val > 10 else red

                plot_text(
                    scr,
                    bat_string,
                    bat_col,
                    16,
                    reso,
                    current_states[agent]["xy_pos"][0],
                    current_states[agent]["xy_pos"][1],
                    min_x,
                    max_y,
                    x_range,
                    y_range,
                    0,
                    15,
                )
            elif current_states[agent].get("type") == "tug":
                # Show only tug ID when empty
                id_string = f"TUG: {current_states[agent].get('id')}"
                col = blue

                # New Battery Line
                is_charging = current_states[agent].get("is_charging", False)
                # We use int() to keep the display clean, or :.1f if you want one decimal place
                battery_val = current_states[agent].get("battery", 999)
                # Update label to show (+) and use Green when charging
                if is_charging:
                    bat_string = f"BAT: {int(battery_val)}% (+)"
                    bat_col = lightgreen  # Bright green for charging
                else:
                    bat_string = f"BAT: {int(battery_val)}%"
                    # color based on battery health
                    # Green (>30), Yellow (10-30), or Red (<10)
                    bat_col = green if battery_val > 30 else yellow if battery_val > 10 else red

                plot_text(
                    scr,
                    bat_string,
                    bat_col,
                    16,
                    reso,
                    current_states[agent]["xy_pos"][0],
                    current_states[agent]["xy_pos"][1],
                    min_x,
                    max_y,
                    x_range,
                    y_range,
                    0,
                    15,
                )
            else:
                # Show only aircraft ID for independent aircraft
                ac_id = current_states[agent].get("ac_id", current_states[agent].get("id"))
                id_string = f"AC: {ac_id}"
                col = red

            plot_text(
                scr,
                id_string,
                col,
                14,
                reso,
                current_states[agent]["xy_pos"][0],
                current_states[agent]["xy_pos"][1],
                min_x,
                max_y,
                x_range,
                y_range,
                0,
                25,
            )

    if disp_time:
        plot_text(
            scr,
            "timestep",
            black,
            30,
            reso,
            min_x + 0.90 * x_range,
            max_y - 0.03 * y_range,
            min_x,
            max_y,
            x_range,
            y_range,
        )
        time = t
        plot_text(
            scr,
            str(time).zfill(2),
            black,
            25,
            reso,
            min_x + 0.90 * x_range,
            max_y - 0.06 * y_range,
            min_x,
            max_y,
            x_range,
            y_range,
        )

    collision = False
    for ac1 in current_states:
        for ac2 in current_states:
            # ---> MODIFIED: Check that BOTH agents are Tugs before declaring a collision <---
            if (
                ac1 != ac2
                and current_states[ac1].get("type") in ["tug", "tug_towing"]
                and current_states[ac2].get("type") in ["tug", "tug_towing"]
                and current_states[ac1]["xy_pos"] == current_states[ac2]["xy_pos"]
            ):

                collision = True

                # Safely get IDs for both agents
                id1 = current_states[ac1].get("id")
                id2 = current_states[ac2].get("id")

                print(
                    "COLLISION - between TUG",
                    id1,
                    "and TUG",
                    id2,
                    "at location",
                    current_states[ac1]["xy_pos"],
                    "time",
                    time,
                )
                plot_text(
                    scr,
                    "COLLISION",
                    purple,
                    16,
                    reso,
                    current_states[ac1]["xy_pos"][0],
                    current_states[ac1]["xy_pos"][1] + 0.1,
                    min_x,
                    max_y,
                    x_range,
                    y_range,
                    0,
                    25,
                )

    pg.display.flip()  # Update the full display Surface to the screen
    pg.event.pump()  # internally process pygame event handlers
    keys = pg.key.get_pressed()  # get the state of all keyboard buttons

    if collision:
        timer.sleep(0.4)

    if keys[pg.K_ESCAPE]:  # if the escape key is being pressed
        escape_pressed = True  # stop running
        print("Visualization aborted by escape key")
        pg.quit()
    else:
        escape_pressed = False

    if keys[pg.K_p]:
        input("Paused, press enter to continue")

    return escape_pressed
