#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on May 12, 2026, at 14:03
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = '2024_RDpaindiscrimination_QUEST'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': 'sub-000',
    'com_thermode': 'COM3',
    'temp_plateau': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920,1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s/QUEST/%s_%s_%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'] )
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\mplab\\Desktop\\RD_2024\\02_2024_RDpaindiscrimination_QUEST_v5.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color='black', colorSpace='named',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'black'
        win.colorSpace = 'named'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from init_thermode
    
    
    import pandas as pd
    # Initialize QUEST
    import numpy as np
    import random
    import threading
    from psychopy import data
    from questplus.psychometric_function import weibull
    
    # Thermode
    from pytcsii import tcsii_serial
    port_thermode = tcsii_serial(str(expInfo['com_thermode']), beep=False, temp_profile=True)
    port_thermode.set_baseline(38)
    port_thermode.port.write('Ue11111'.encode()) # Enable temp profile
    
    # Get the plateau temp from the calibration
    temp_plateau = float(expInfo['temp_plateau'])
    
    # Stimulus domain.
    intensities = np.arange(0.2, 1.2, 0.1) # Difference from plateau
    thresholds = intensities.copy()
    slopes = np.linspace(0.1, 15, 100)
    lapses = np.arange(0., 0.02, 0.01)
    
    # Experiment parameters
    num_trials_max = 40
    
    # create  "staircase"
    staircase = data.StairHandler(startVal=0.8,
                                  stepType='lin',
                                  stepSizes = [0.4, 0.3, 0.2, 0.1],
                                  nUp=1,
                                  minVal=0.3,
                                  maxVal=1,
                                  nDown=2,
                                  nTrials=num_trials_max)
    
    baseline = 38
    rise_time = .75
    
    rise_plateau = np.round((temp_plateau - baseline)/rise_time, 1)
    
    part = int(expInfo['participant'][4:])-1
    
    # Part pair ou impair : Sélection couleur
    col1 = [1, 0.5059, -1]
    col2 = [-1, 0.38, 0.88]
    col3 = [-0.5, -0.5, -0.5]
    if part % 2 == 0:
        #Triangle bleu, carré jaune
        color_inactif_trig = col1
        color_actif_carre = col2
        #texte oui-non
        loca_textl = 'OUI              '
        loca_textr = '              NON'
        detectl = 1
        detectr = 0
    
        print('impair')
    else:
        #Triangle jaune, carrée bleu
        color_inactif_trig = col2
        color_actif_carre = col1
        print('pair')
        #texte oui-non
        loca_textl = 'NON              '
        loca_textr = '              OUI'
        detectl = 0
        detectr = 1
    
    
    #Polygon color
    polygon_1 = visual.Circle(
        win=win, name='polygon',
        size=(0.5, 0.5), 
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor= 'white', fillColor= 'white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    #Black polygon
    
    polygon_1_black = visual.Circle(
        win=win, name='polygon_1_black',
        size=(0.4, 0.4),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor= col3, fillColor= col3,
        opacity=1.0, depth=-1.0, interpolate=True)
    
    
    # Font size
    cross_size = 0.2
    loctherm_size = 0.02
    
    curr_item = -1
    count_trials = -1
    text_bonjour = visual.TextStim(win=win, name='text_bonjour',
        text="Bienvenue dans le programme de calibration de discrimination.\n\nSi vous ressentez une différence de chaleur pendant la stimulation, \nveuillez sélectionner OUI ou NON à l'aide de la boîte de commande.\n\nExpérimentateur.trice, appuyer sur 'p' démarrer l'expérience\n\nBonne expérience.\n",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    initialisation = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "wait" ---
    text_pic = visual.TextStim(win=win, name='text_pic',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), draggable=False, height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    croix_3 = visual.TextStim(win=win, name='croix_3',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=cross_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "triggerrrs" ---
    # Run 'Begin Experiment' code from trigger_pic_2
    intensities_all = []
    responses_all = []
    
    thermode_locali_pic = visual.TextStim(win=win, name='thermode_locali_pic',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), draggable=False, height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    discrimin_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    responsel = visual.TextStim(win=win, name='responsel',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    percu_or = visual.TextStim(win=win, name='percu_or',
        text='Avez-vous perçu un changement?',
        font='Open Sans',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    responser = visual.TextStim(win=win, name='responser',
        text=loca_textr,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "Cali_fin" ---
    text = visual.TextStim(win=win, name='text',
        text='Calibration terminée.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "trial" ---
    # create an object to store info about Routine trial
    trial = data.Routine(
        name='trial',
        components=[text_bonjour, initialisation],
    )
    trial.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from init_thermode
    win.mouseVisible = False
    # create starting attributes for initialisation
    initialisation.keys = []
    initialisation.rt = []
    _initialisation_allKeys = []
    # store start times for trial
    trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    trial.tStart = globalClock.getTime(format='float')
    trial.status = STARTED
    thisExp.addData('trial.started', trial.tStart)
    trial.maxDuration = None
    # keep track of which components have finished
    trialComponents = trial.components
    for thisComponent in trial.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial" ---
    thisExp.currentRoutine = trial
    trial.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_bonjour* updates
        
        # if text_bonjour is starting this frame...
        if text_bonjour.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_bonjour.frameNStart = frameN  # exact frame index
            text_bonjour.tStart = t  # local t and not account for scr refresh
            text_bonjour.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_bonjour, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_bonjour.status = STARTED
            text_bonjour.setAutoDraw(True)
        
        # if text_bonjour is active this frame...
        if text_bonjour.status == STARTED:
            # update params
            pass
        
        # *initialisation* updates
        waitOnFlip = False
        
        # if initialisation is starting this frame...
        if initialisation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            initialisation.frameNStart = frameN  # exact frame index
            initialisation.tStart = t  # local t and not account for scr refresh
            initialisation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(initialisation, 'tStartRefresh')  # time at next scr refresh
            # update status
            initialisation.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(initialisation.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(initialisation.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if initialisation.status == STARTED and not waitOnFlip:
            theseKeys = initialisation.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _initialisation_allKeys.extend(theseKeys)
            if len(_initialisation_allKeys):
                initialisation.keys = _initialisation_allKeys[-1].name  # just the last key pressed
                initialisation.rt = _initialisation_allKeys[-1].rt
                initialisation.duration = _initialisation_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=trial,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            trial.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if trial.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in trial.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial" ---
    for thisComponent in trial.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for trial
    trial.tStop = globalClock.getTime(format='float')
    trial.tStopRefresh = tThisFlipGlobal
    thisExp.addData('trial.stopped', trial.tStop)
    thisExp.nextEntry()
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=num_trials_max, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "wait" ---
        # create an object to store info about Routine wait
        wait = data.Routine(
            name='wait',
            components=[text_pic, croix_3],
        )
        wait.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from get_list_discri_2
        
        # Counter for thermode loc
        curr_item += 1
        thermode_loca = [0, 1, 2, 3]*12
        thermode_localisation = 'T : ' + str(thermode_loca[curr_item]) 
        
        
        # store start times for wait
        wait.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wait.tStart = globalClock.getTime(format='float')
        wait.status = STARTED
        thisExp.addData('wait.started', wait.tStart)
        wait.maxDuration = None
        # keep track of which components have finished
        waitComponents = wait.components
        for thisComponent in wait.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "wait" ---
        thisExp.currentRoutine = wait
        wait.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_pic* updates
            
            # if text_pic is starting this frame...
            if text_pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_pic.frameNStart = frameN  # exact frame index
                text_pic.tStart = t  # local t and not account for scr refresh
                text_pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_pic, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_pic.status = STARTED
                text_pic.setAutoDraw(True)
            
            # if text_pic is active this frame...
            if text_pic.status == STARTED:
                # update params
                text_pic.setText(thermode_localisation, log=False)
            
            # if text_pic is stopping this frame...
            if text_pic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_pic.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_pic.tStop = t  # not accounting for scr refresh
                    text_pic.tStopRefresh = tThisFlipGlobal  # on global time
                    text_pic.frameNStop = frameN  # exact frame index
                    # update status
                    text_pic.status = FINISHED
                    text_pic.setAutoDraw(False)
            
            # *croix_3* updates
            
            # if croix_3 is starting this frame...
            if croix_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                croix_3.frameNStart = frameN  # exact frame index
                croix_3.tStart = t  # local t and not account for scr refresh
                croix_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(croix_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                croix_3.status = STARTED
                croix_3.setAutoDraw(True)
            
            # if croix_3 is active this frame...
            if croix_3.status == STARTED:
                # update params
                pass
            
            # if croix_3 is stopping this frame...
            if croix_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > croix_3.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    croix_3.tStop = t  # not accounting for scr refresh
                    croix_3.tStopRefresh = tThisFlipGlobal  # on global time
                    croix_3.frameNStop = frameN  # exact frame index
                    # update status
                    croix_3.status = FINISHED
                    croix_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=wait,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                wait.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if wait.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in wait.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wait" ---
        for thisComponent in wait.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wait
        wait.tStop = globalClock.getTime(format='float')
        wait.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wait.stopped', wait.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if wait.maxDurationReached:
            routineTimer.addTime(-wait.maxDuration)
        elif wait.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # --- Prepare to start Routine "triggerrrs" ---
        # create an object to store info about Routine triggerrrs
        triggerrrs = data.Routine(
            name='triggerrrs',
            components=[thermode_locali_pic, discrimin_resp, responsel, percu_or, responser],
        )
        triggerrrs.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from trigger_pic_2
        
        # Get pic stim
        questpic = staircase.next()
        pic = temp_plateau + questpic
        
        # Convert to str to send to thermode
        pic_str = str(int(np.round(pic, 1)*10)) 
        temp_plateau_str = str(int(np.round(temp_plateau, 1)*10))
        
        
        response_pic = None
        # Jitter duration
        dur_plateau_1 = round(random.uniform(3.25, 4.75), 2)
        
        # Plateau 2 is difference 10 - others
        dur_plateau_2 = round(10 - (dur_plateau_1 + 1.2), 2)
        
        # Detection cue always start at 3.5
        start1 = 3
        # Responses start after rise to plateau + plateau + pic (1.2)
        start2 = 8
        
        # Convert to string
        dur_send = str(int(dur_plateau_1*100))
        dur_send2 = str(int(dur_plateau_2*100))
        
        #Pic or not?
            # Set thermode
        port_thermode.set_rd_plateau(temp_plateau=temp_plateau_str,
                          temp_pic=pic_str,
                          dur_plateau_1_10ms=dur_send,
                          dur_plateau_2_10ms=dur_send2)
        
        ## Trigger thermode in other thread
        out_file = u'data/%s_%s/QUEST/%s_%s_%s%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], expInfo['participant'] + '_temp_trial_' + str(trials.thisN).zfill(2) + '.csv')
        stim_thread = threading.Thread(target=port_thermode.trigger_and_save_temp_rd,
                             args=(out_file, 11500,))
        stim_thread.start()
        
        
        
        
        
        
        
        thermode_locali_pic.setText(thermode_localisation)
        # create starting attributes for discrimin_resp
        discrimin_resp.keys = []
        discrimin_resp.rt = []
        _discrimin_resp_allKeys = []
        responsel.setText(loca_textl)
        # store start times for triggerrrs
        triggerrrs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        triggerrrs.tStart = globalClock.getTime(format='float')
        triggerrrs.status = STARTED
        thisExp.addData('triggerrrs.started', triggerrrs.tStart)
        triggerrrs.maxDuration = None
        # keep track of which components have finished
        triggerrrsComponents = triggerrrs.components
        for thisComponent in triggerrrs.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "triggerrrs" ---
        thisExp.currentRoutine = triggerrrs
        triggerrrs.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from trigger_pic_2
            #Polygon normal avant signal
            if (t <= start1): 
                polygon_1.draw()
            
            #Remplir en noir pour signal
            if(t >= start1) and (t <= start2): 
                polygon_1.draw()
                polygon_1_black.draw()
            
            # Check response and change color answer
            if discrimin_resp.status == STARTED and len(discrimin_resp.keys) != 0:
                if discrimin_resp.keys[-1] == 'm':
                    responsel.color = (0, 0, 0)
                    response_pic = detectr
                elif discrimin_resp.keys[-1] == 'n':
                    responser.color = (0, 0, 0)
                    response_pic = detectl
            
            #if (t >= 7.90) :
                #win.flip()
            
            # *thermode_locali_pic* updates
            
            # if thermode_locali_pic is starting this frame...
            if thermode_locali_pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                thermode_locali_pic.frameNStart = frameN  # exact frame index
                thermode_locali_pic.tStart = t  # local t and not account for scr refresh
                thermode_locali_pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(thermode_locali_pic, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thermode_locali_pic.started')
                # update status
                thermode_locali_pic.status = STARTED
                thermode_locali_pic.setAutoDraw(True)
            
            # if thermode_locali_pic is active this frame...
            if thermode_locali_pic.status == STARTED:
                # update params
                pass
            
            # if thermode_locali_pic is stopping this frame...
            if thermode_locali_pic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > thermode_locali_pic.tStartRefresh + 10.2-frameTolerance:
                    # keep track of stop time/frame for later
                    thermode_locali_pic.tStop = t  # not accounting for scr refresh
                    thermode_locali_pic.tStopRefresh = tThisFlipGlobal  # on global time
                    thermode_locali_pic.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'thermode_locali_pic.stopped')
                    # update status
                    thermode_locali_pic.status = FINISHED
                    thermode_locali_pic.setAutoDraw(False)
            
            # *discrimin_resp* updates
            waitOnFlip = False
            
            # if discrimin_resp is starting this frame...
            if discrimin_resp.status == NOT_STARTED and tThisFlip >= start2-frameTolerance:
                # keep track of start time/frame for later
                discrimin_resp.frameNStart = frameN  # exact frame index
                discrimin_resp.tStart = t  # local t and not account for scr refresh
                discrimin_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(discrimin_resp, 'tStartRefresh')  # time at next scr refresh
                # update status
                discrimin_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(discrimin_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(discrimin_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if discrimin_resp is stopping this frame...
            if discrimin_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > discrimin_resp.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    discrimin_resp.tStop = t  # not accounting for scr refresh
                    discrimin_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    discrimin_resp.frameNStop = frameN  # exact frame index
                    # update status
                    discrimin_resp.status = FINISHED
                    discrimin_resp.status = FINISHED
            if discrimin_resp.status == STARTED and not waitOnFlip:
                theseKeys = discrimin_resp.getKeys(keyList=['n', 'm'], ignoreKeys=["escape"], waitRelease=False)
                _discrimin_resp_allKeys.extend(theseKeys)
                if len(_discrimin_resp_allKeys):
                    discrimin_resp.keys = _discrimin_resp_allKeys[-1].name  # just the last key pressed
                    discrimin_resp.rt = _discrimin_resp_allKeys[-1].rt
                    discrimin_resp.duration = _discrimin_resp_allKeys[-1].duration
            
            # *responsel* updates
            
            # if responsel is starting this frame...
            if responsel.status == NOT_STARTED and tThisFlip >= start2-frameTolerance:
                # keep track of start time/frame for later
                responsel.frameNStart = frameN  # exact frame index
                responsel.tStart = t  # local t and not account for scr refresh
                responsel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(responsel, 'tStartRefresh')  # time at next scr refresh
                # update status
                responsel.status = STARTED
                responsel.setAutoDraw(True)
            
            # if responsel is active this frame...
            if responsel.status == STARTED:
                # update params
                pass
            
            # if responsel is stopping this frame...
            if responsel.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > responsel.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    responsel.tStop = t  # not accounting for scr refresh
                    responsel.tStopRefresh = tThisFlipGlobal  # on global time
                    responsel.frameNStop = frameN  # exact frame index
                    # update status
                    responsel.status = FINISHED
                    responsel.setAutoDraw(False)
            
            # *percu_or* updates
            
            # if percu_or is starting this frame...
            if percu_or.status == NOT_STARTED and tThisFlip >= start2-frameTolerance:
                # keep track of start time/frame for later
                percu_or.frameNStart = frameN  # exact frame index
                percu_or.tStart = t  # local t and not account for scr refresh
                percu_or.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(percu_or, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'percu_or.started')
                # update status
                percu_or.status = STARTED
                percu_or.setAutoDraw(True)
            
            # if percu_or is active this frame...
            if percu_or.status == STARTED:
                # update params
                pass
            
            # if percu_or is stopping this frame...
            if percu_or.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > percu_or.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    percu_or.tStop = t  # not accounting for scr refresh
                    percu_or.tStopRefresh = tThisFlipGlobal  # on global time
                    percu_or.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'percu_or.stopped')
                    # update status
                    percu_or.status = FINISHED
                    percu_or.setAutoDraw(False)
            
            # *responser* updates
            
            # if responser is starting this frame...
            if responser.status == NOT_STARTED and tThisFlip >= start2-frameTolerance:
                # keep track of start time/frame for later
                responser.frameNStart = frameN  # exact frame index
                responser.tStart = t  # local t and not account for scr refresh
                responser.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(responser, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'responser.started')
                # update status
                responser.status = STARTED
                responser.setAutoDraw(True)
            
            # if responser is active this frame...
            if responser.status == STARTED:
                # update params
                pass
            
            # if responser is stopping this frame...
            if responser.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > responser.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    responser.tStop = t  # not accounting for scr refresh
                    responser.tStopRefresh = tThisFlipGlobal  # on global time
                    responser.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'responser.stopped')
                    # update status
                    responser.status = FINISHED
                    responser.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=triggerrrs,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                triggerrrs.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if triggerrrs.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in triggerrrs.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "triggerrrs" ---
        for thisComponent in triggerrrs.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for triggerrrs
        triggerrrs.tStop = globalClock.getTime(format='float')
        triggerrrs.tStopRefresh = tThisFlipGlobal
        thisExp.addData('triggerrrs.stopped', triggerrrs.tStop)
        # Run 'End Routine' code from trigger_pic_2
        trials.addData('dur_plateau1', dur_plateau_1)
        trials.addData('dur_plateau2', dur_plateau_2)
        trials.addData('dur_plateau1_sent', dur_send)
        trials.addData('dur_plateau2_sent', dur_send2)
        trials.addData('start_plateau_2_resp', start2)
        trials.addData('start_detect_cue', start1)
        trials.addData('dur_pic', 1.2)
        trials.addData('dur_rise', 0.75)
        trials.addData('temp_plateau_sent', temp_plateau_str)
        
        trials.addData('loca_thermode', thermode_loca[curr_item])
        trials.addData('pic_sent', pic_str)
        trials.addData('quest_intens', questpic)
        trials.addData('pic_response', response_pic)
        # Reset color
        responsel.color = 'white'
        responser.color = 'white'
        
        
        
        if response_pic == 1:
            outcome = 1
            staircase.addResponse(outcome, questpic)
            count_trials +=1
            intensities_all.append(questpic)
            responses_all.append(outcome)
        elif response_pic == 0:
            outcome = 0
            staircase.addResponse(outcome, questpic)
            count_trials +=1
            intensities_all.append(questpic)
            responses_all.append(outcome)
        else:
            pass
        
        
        if count_trials == 31:
            trials.finished = 1
        
        # check responses
        if discrimin_resp.keys in ['', [], None]:  # No response was made
            discrimin_resp.keys = None
        trials.addData('discrimin_resp.keys',discrimin_resp.keys)
        if discrimin_resp.keys != None:  # we had a response
            trials.addData('discrimin_resp.rt', discrimin_resp.rt)
            trials.addData('discrimin_resp.duration', discrimin_resp.duration)
        # the Routine "triggerrrs" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed num_trials_max repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Cali_fin" ---
    # create an object to store info about Routine Cali_fin
    Cali_fin = data.Routine(
        name='Cali_fin',
        components=[text],
    )
    Cali_fin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Cali_fin
    Cali_fin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Cali_fin.tStart = globalClock.getTime(format='float')
    Cali_fin.status = STARTED
    thisExp.addData('Cali_fin.started', Cali_fin.tStart)
    Cali_fin.maxDuration = None
    # keep track of which components have finished
    Cali_finComponents = Cali_fin.components
    for thisComponent in Cali_fin.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Cali_fin" ---
    thisExp.currentRoutine = Cali_fin
    Cali_fin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 6.0-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.tStopRefresh = tThisFlipGlobal  # on global time
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Cali_fin,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Cali_fin.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Cali_fin.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Cali_fin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Cali_fin" ---
    for thisComponent in Cali_fin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Cali_fin
    Cali_fin.tStop = globalClock.getTime(format='float')
    Cali_fin.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Cali_fin.stopped', Cali_fin.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Cali_fin.maxDurationReached:
        routineTimer.addTime(-Cali_fin.maxDuration)
    elif Cali_fin.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from trigger_pic_2
    #print(f'\nParameter estimates:\n')
    #for param_name, value in staircase.paramEstimate.items():
    #    print(f'    {param_name}: {value:.3f}')
    #    thisExp.addData(param_name, value)
    #
    #pic_a_utiliser = staircase.paramEstimate['threshold'] + temp_plateau
    #thisExp.addData('pic_a_utiliser', pic_a_utiliser)
    
    
    print('mean of final 6 reversals = %.3f' %(np.average(staircase.reversalIntensities[-6:])))
    thresh = np.average(staircase.reversalIntensities[-6:])
    thisExp.addData('mean_6_reversals', thresh)
    thisExp.addData('pic_a_utiliser', temp_plateau + thresh)
    
    
    try: 
        import matplotlib.pyplot as plt
        quest_intensities = intensities_all
        quest_detections = responses_all
    
        fit = data.FitWeibull(quest_intensities, quest_detections, expectedMin=0,
                                guess=[0.1, 0.5])
        smoothInt = np.arange(min(quest_intensities), max(quest_detections), 0.001)
        smoothResp = fit.eval(smoothInt)
        # thresh = staircase.paramEstimate['threshold']
        thresh_final = fit.inverse(0.75)
        print(thresh_final)
        thisExp.addData('threshold', thresh_final)
    
    
        plt.figure(figsize=(10,5))
        plt.subplot(122)
        plt.plot(smoothInt, smoothResp, '-')
        # plt.title('threshold = %0.3f' %(thresh))
        plt.axvline(x=thresh_final, color='k', linestyle='--')
        #plot points
        plt.plot(quest_intensities, quest_detections, 'o')
        plt.ylim([-0.5,1.5])
        out_file = u'data/%s_%s/QUEST/%s_%s_%s%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], expInfo['participant'] + '_staircase.png')
        plt.savefig(out_file)
    except:
        thisExp.addData('threshold', 'failed')
        print('Fitting weibull failed')
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
