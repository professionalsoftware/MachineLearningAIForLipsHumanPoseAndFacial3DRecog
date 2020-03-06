#!/usr/bin/python3
 
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from argparse import ArgumentParser
import lipsbodypose
import cv2
import numpy
import math
import sys

from facial_landmarks import FacialLandmarksEngine

class PoseRenderConstants:
    PartIdx = {
        'NOSE': 0,
        'NECK': 1,
        'R_SHOULDER': 2,
        'R_ELBOW': 3,
        'R_WRIST': 4,
        'L_SHOULDER': 5, 
        'L_ELBOW': 6,
        'L_WRIST': 7,
        'R_HIP': 8,
        'R_KNEE': 9,
        'R_ANKLE': 10,
        'L_HIP': 11,
        'L_KNEE': 12,
        'L_ANKLE': 13,
        'R_EYE': 14,
        'L_EYE': 15,
        'R_EAR': 16,
        'L_EAR': 17
    }
    
    # color: 0~1
    PartPairs = (
        #Face
        { 'pair': ( 'R_EYE', 'R_EAR' ), 'color': ( 1.0, 0.0, 0.0 ) },
        { 'pair': ( 'L_EYE', 'L_EAR' ), 'color': ( 1.0, 0.0, 0.0 ) },
        { 'pair': ( 'L_EYE', 'R_EYE' ), 'color': ( 1.0, 0.0, 0.0 ) },
        { 'pair': ( 'L_EAR', 'NECK' ),  'color': ( 1.0, 0.0, 0.0 ) },
        { 'pair': ( 'R_EAR', 'NECK' ),  'color': ( 1.0, 0.0, 0.0 ) },
        #Body
        { 'pair': ( 'NECK', 'R_SHOULDER' ),  'color': ( 0.0, 1.0, 0.0 ) },
        { 'pair': ( 'R_SHOULDER', 'R_HIP' ), 'color': ( 0.0, 1.0, 0.0 ) },
        { 'pair': ( 'NECK', 'L_SHOULDER' ),  'color': ( 0.0, 1.0, 0.0 ) },
        { 'pair': ( 'L_SHOULDER', 'L_HIP' ), 'color': ( 0.0, 1.0, 0.0 ) },
        { 'pair': ( 'L_HIP', 'R_HIP' ),      'color': ( 0.0, 1.0, 0.0 ) },
        #Right arm
        { 'pair': ( 'R_SHOULDER', 'R_ELBOW' ), 'color': ( 0.0, 0.5, 1.0 ) },
        { 'pair': ( 'R_ELBOW', 'R_WRIST' ),    'color': ( 0.0, 0.5, 1.0 ) },
        #Left arm
        { 'pair': ( 'L_SHOULDER', 'L_ELBOW' ), 'color': ( 1.0, 0.0, 0.5 ) },
        { 'pair': ( 'L_ELBOW', 'L_WRIST' ),    'color': ( 1.0, 0.0, 0.5 ) },
        #Right leg
        { 'pair': ( 'R_HIP', 'R_KNEE' ),   'color': ( 0.0, 1.0, 0.5 ) },
        { 'pair': ( 'R_KNEE', 'R_ANKLE' ), 'color': ( 0.0, 1.0, 0.5 ) },
        #Left leg
        { 'pair': ( 'L_HIP', 'L_KNEE' ),   'color': ( 0.5, 0.0, 1.0 ) },
        { 'pair': ( 'L_KNEE', 'L_ANKLE' ), 'color': ( 0.5, 0.0, 1.0 ) }
    )
    
    # color: 0~255
    MaskColors = (
        (   0,   0, 127 ),
        (   0, 127, 127 ),
        ( 127,   0, 127 ),
        ( 127, 127,   0 ),
        ( 127,   0, 127 )
    )


    
# Initial global variable ############################################################################

mat_ambient = ( 0.8, 0.8, 0.8, 1.0 )
mat_specular = ( 1.0, 1.0, 1.0, 1.0 )
mat_shininess = ( 50.0 )
light_position = ( 0.0, 0.0, 0.0, 1.0 )

quadric = gluNewQuadric()

rgb   = numpy.empty( 0, dtype=numpy.uint8 )
depth = numpy.empty( 0, dtype=numpy.uint16 )

skeleton2d = []
skeleton3d = []
humanId = []

win3d = []
win2d = []

tex_inited = False

pose = lipsbodypose.lipsbodypose()

facialLandmarksEngineObj = None

# timer ############################################################################
def timer(value):
    global win3d
    global win2d

    global rgb
    global depth
    global skeleton2d
    global skeleton3d
    global humanId

    # 3D #=====================================================
    glutSetWindow(win3d)
    ( rgb, depth, skeleton2d, skeleton3d, humanId) = pose.readFrame()
    glutPostRedisplay()

    # 2D #=====================================================
    glutSetWindow(win2d)
    glutPostRedisplay()

    # Process frame for facial landmarks recognition
    facialLandmarksEngineObj.process_frame(rgb)

    glutTimerFunc(0, timer, 0)


def Idle():
    return

# object ###########################################################################

def draw3DLine( pointA, pointB, width = 0.4, slices = 10 ):
    global quadric
    d = numpy.subtract( pointB, pointA )
    z = [ 0, 0, 1 ]

    length = math.sqrt( numpy.dot( d, d ) )
    
    if length > 0:
        angle = math.acos( d[2] / length)
        z = numpy.cross( z, d )

        glPushMatrix()
        glTranslatef( pointA[0], pointA[1], pointA[2] )
        glRotatef( angle * 180 / math.pi, z[0], z[1], z[2] )
        gluCylinder( quadric, width / 2, width / 2, length, slices, 1 )
        glPopMatrix()

def setDiffuseColor( color ):
    mat_diffuse = ( color[0], color[1], color[2], 1.0 )
    glMaterialfv( GL_FRONT, GL_DIFFUSE, mat_diffuse )

def renderPose3D():
    global rgb
    global depth
    global skeleton2d
    global skeleton3d
    coord_factor = 0.005
    #scaled_3m = 3000 * coord_factor
    for person in skeleton3d:
        for item in PoseRenderConstants.PartPairs:
            ( partA, partB ) = item['pair']
            color = item['color']
                
            keypointA = person[PoseRenderConstants.PartIdx[partA]]
            keypointB = person[PoseRenderConstants.PartIdx[partB]]
            keypointA = [ element*coord_factor for element in keypointA ] #[xc, yc, zc] = (x, y, z) * coord_factor
            keypointB = [ element*coord_factor for element in keypointB ]

            zA = keypointA[2]
            zB = keypointB[2]                
            #if 0 <= zA < scaled_3m and 0 <= zB < scaled_3m: #z = [0m, 3m)
            if zA >= 0 and zB >= 0:
                setDiffuseColor( color )
                draw3DLine( keypointA, keypointB ) #normal skeleton

        for keypoint in person:
            keypoint = [ element*coord_factor for element in keypoint ] #[xc, yc, zc] = (x, y, z) * coord_factor

            z = keypoint[2]
            #if 0 <= z < scaled_3m: #0~3m
            if z >= 0:
                glPushMatrix()
                setDiffuseColor( ( 1, 0.5, 0.5 ) )
                glTranslatef( keypoint[0], keypoint[1], keypoint[2] ) #glutSolidSpere draw at (0,0,0), so translate to target position and draw
                glutSolidSphere( 0.3, 10, 10 ) # normal skeleton
                glPopMatrix()

def draw_object_3d():

    glLoadIdentity()

    scale = 0.5
    #scale = 1
    glScalef( scale, scale, scale )

    glEnable( GL_LIGHTING )
    renderPose3D()
    glDisable( GL_LIGHTING )


def renderTex():
    glEnable( GL_TEXTURE_2D )

    glColor4f( 1.0, 1.0, 1.0, 1.0 )
    glBegin( GL_QUADS )

    glTexCoord2f( 0.0, 0.0 )
    glVertex2f( 0.0, 0.0 )
    glTexCoord2f( 0.0, 1.0 )
    glVertex2f( 0.0, 1.0 )
    glTexCoord2f( 1.0, 1.0 )
    glVertex2f( 1.0, 1.0 )
    glTexCoord2f( 1.0, 0.0 )
    glVertex2f( 1.0, 0.0 )

    glEnd()

    glDisable( GL_TEXTURE_2D )

def drawFilledCircle( position, radius, triangleAmount ):
    twicePi = 2.0 * math.pi
    glBegin( GL_TRIANGLE_FAN )
    glVertex2fv( position ) # center of circle
    for i in range( triangleAmount + 1 ):
        glVertex2f(
            position[0] + ( radius * math.cos( i * twicePi / triangleAmount ) ),
            position[1] + ( radius * math.sin( i * twicePi / triangleAmount ) )
        )
    glEnd()

def renderPose2D():
    global rgb
    global depth
    global skeleton2d
    global skeleton3d
    global humanId

    person_id = 0
    for person in skeleton2d:
        
        #=============================================================================================================
        glLineWidth( 3 ) #seting pair line width
        for item in PoseRenderConstants.PartPairs:
            ( partA, partB ) = item['pair']

            keypointA = person[PoseRenderConstants.PartIdx[partA]]
            keypointB = person[PoseRenderConstants.PartIdx[partB]]
        
            if all( element >= 0 for element in keypointA+keypointB): #check A and B all element >= 0
                glColor3f( 1.0, 1.0, 1.0 )
                glBegin( GL_LINES )
                glVertex2fv( keypointA )
                glVertex2fv( keypointB )
                glEnd()

        for keypoint in person:
            if all( element >= 0 for element in keypoint ):
                glColor3f( 1, 0.2, 0.2 ) #pink
                drawFilledCircle( keypoint, 0.007, 10 )

        #render neck distance
        if depth.size > 0:
            keypoint = person[PoseRenderConstants.PartIdx['NECK']]
            if all( 0 <= element < 1 for element in keypoint ):
                ( height, width ) = depth.shape[:2]
                ( x, y ) = keypoint

                position = ( int(y * height), int(x * width))
                distance = 0.001 * depth.item( position )

                if distance > 0:
                    #text = "{:.3f}".format( distance )
                    text = "{:d}".format( humanId[person_id] )

                    glPushMatrix()

                    glTranslatef( x-0.1, y, 0 )
                    glScalef( 0.0006, 0.0006, 1 )
                    gluOrtho2D( 0.0, 1.0, 1.0, 0.0 ) #flip vertically

                    glLineWidth( 20 )
                    #glColor3f( 0, 0, 0 )
                    glColor3f( 1, 1, 1 )
                    
                    glutStrokeString( GLUT_STROKE_ROMAN, bytes( text, 'utf-8' ) ) #convert string to bytes for c func
                    glPopMatrix()

        person_id += 1
        #=============================================================================================================

def getHumanSegmentation():
    global rgb
    global depth
    global skeleton2d
    global skeleton3d
    mask_dic = {}

    if depth.size > 0:
        ( height, width ) = depth.shape[:2]

        for person_id, person in enumerate( skeleton2d ):

            #print("Hello_getHumanSegmentation")

            ankle_l_x = person[PoseRenderConstants.PartIdx['L_ANKLE']][0]
            ankle_r_x = person[PoseRenderConstants.PartIdx['R_ANKLE']][0]
            ankle_exist  = ( ankle_l_x > 0 ) or ( ankle_r_x > 0 )

            wrist_l_x = person[PoseRenderConstants.PartIdx['L_WRIST']][0]
            wrist_r_x = person[PoseRenderConstants.PartIdx['R_WRIST']][0]
            both_wrists_exist = ( wrist_l_x > 0 ) and ( wrist_r_x > 0 )

            if not( ankle_exist and both_wrists_exist ):
                continue
           
            ( min_x, min_y ) = ( width, height )
            ( max_x, max_y ) = ( 0, 0 )
            ( min_z, max_z ) = ( 65535, 0 )
            for part_id, keypoint in enumerate(person):
                if all( element >= 0 for element in keypoint ):
                    ( x, y ) = keypoint
                    z = skeleton3d[person_id][part_id][2]
                    min_x = x if ( x < min_x ) else min_x
                    min_y = y if ( y < min_y ) else min_y
                    max_x = x if ( x > max_x ) else max_x
                    max_y = y if ( y > max_y ) else max_y
                    
                    if z > 0 :
                        min_z = z if ( z < min_z ) else min_z
                        max_z = z if ( z > max_z ) else max_z

            padding_x      = ( max_x - min_x ) * 0.2
            padding_top    = ( max_y - min_y ) * 0.4
            padding_bottom = 0
            padding_z      = 300

            #normalization and padding
            min_x = int( ( min_x - padding_x ) * width )
            max_x = int( ( max_x + padding_x ) * width )
            min_y = int( ( min_y - padding_top ) * height )
            max_y = int( ( max_y + padding_bottom ) * height )
            min_z = min_z - padding_z
            max_z = max_z + padding_z
    
            bounding_box_mask = numpy.zeros_like(depth, dtype=numpy.uint8)
            cv2.rectangle(bounding_box_mask, ( min_x, min_y ), ( max_x, max_y ), 255, -1)
            depth_mask = ( depth > min_z ) & ( depth < max_z )
            mask_person = ( depth_mask & bounding_box_mask ) > 0

            mask_dic[min_x] = mask_person # use min_x as the key of mask_person

    #sort dictionary by key and convert to list
    mask_list = [ mask_dic[key] for key in sorted( mask_dic.keys() ) ]

    return mask_list

def draw_object_2d():

    global win3d
    global win2d

    global tex_inited

    # 2D #=====================================================
    glutSetWindow(win2d)
    glLoadIdentity()
    
    if depth.size == 0:
        return

    tex_img = ( depth / 8192 * 255 ).astype( 'uint8' )
    tex_img = cv2.applyColorMap( tex_img, cv2.COLORMAP_JET )
    # apply human segmentation coloring
    mask_list = getHumanSegmentation()
    for mask_id, mask in enumerate( mask_list ):
        #get color id by mask id
        color_id = mask_id % len( PoseRenderConstants.MaskColors )
        #blend original image by average width certain color
        tex_img_mask_colored = ( tex_img * ( 0.5, 0.5, 0.5 ) ) + PoseRenderConstants.MaskColors[color_id]
        #copy mask part of colored image to original image
        tex_img[mask] = tex_img_mask_colored[mask]

    ( height, width ) = tex_img.shape[:2]

    if tex_inited:
        glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, tex_img )
    else:
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, tex_img )
        tex_inited = True

    renderTex()

    renderPose2D()


# display ##########################################################################

def display_3d():
    """window redisplay callback."""
    global win3d
    global win2d
    glutSetWindow(win3d)

    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()

    glFrustum( -0.5,  #left,
                0.5,  #right,
                0.5,  #bottom,
               -0.5,  #top,
                  1,  #near,
               1000 ) #far;

    gluLookAt( 0, 0, -10,
               0, 0, 10,
               0, -1, 0 )

    glMatrixMode(GL_MODELVIEW)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    draw_object_3d()

    #glFlush()
    glutSwapBuffers()


def display_2d():
    """window redisplay callback."""
    global win3d
    global win2d
    glutSetWindow(win2d)

    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()
    gluOrtho2D( 0.0, 1.0, 1.0, 0.0 )

    glMatrixMode( GL_MODELVIEW )
    glClear( GL_COLOR_BUFFER_BIT )

    draw_object_2d()

    #glFlush()
    glutSwapBuffers()


# interaction ######################################################################

def keyboard3d(c, x=0, y=0):
    """keyboard callback."""
    if c == b'q':
        glutLeaveMainLoop()
        sys.exit(0)
    glutPostRedisplay()

def keyboard2d(c, x=0, y=0):
    """keyboard callback."""
    if c == b'q':
        glutLeaveMainLoop()
        sys.exit(0)
    glutPostRedisplay()


# setup ############################################################################

WINDOW_SIZE_3D = 640, 640
WINDOW_SIZE_2D = 1280, 720

def init_glut():
    global win3d
    global win2d

    """glut initialization."""
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)

    # win3d =============================
    glutInitWindowSize(*WINDOW_SIZE_3D)
    win3d = glutCreateWindow('Win3D')
    glutDisplayFunc(display_3d)
    glutKeyboardFunc(keyboard3d)

    # win2d =============================
    glutInitWindowSize(*WINDOW_SIZE_2D)
    win2d = glutCreateWindow('Win2D')
    glutDisplayFunc(display_2d)
    glutKeyboardFunc(keyboard2d)


def init_opengl():
    global win3d
    global win2d

    glutSetWindow(win3d)
    # depth test 
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glClearColor( 0.0, 0.0, 0.0, 1.0 )
    glClearDepth( 1.0 )

    # lighting
    glMaterialfv( GL_FRONT, GL_AMBIENT, mat_ambient )
    glMaterialfv( GL_FRONT, GL_SPECULAR, mat_specular )
    glMaterialfv( GL_FRONT, GL_SHININESS, mat_shininess )
    glLightfv( GL_LIGHT0, GL_POSITION, light_position )

    glEnable( GL_LIGHT0 )
    glDisable( GL_LIGHTING )


    glutSetWindow(win2d)
    glActiveTexture( GL_TEXTURE0 )
    glBindTexture( GL_TEXTURE_2D, 0 )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cpu", action="store_true",
                        help="use cpu to do face detection and facial landmark detection",
                        default=False)
    parser.add_argument("--debug", action="store_true",
                        help="show camera image to debug (need to uncomment to show results)",
                        default=False)
    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)
    parser.add_argument('--rotation-to-vertical',
                        help='Optional. rotate camera from horizontal to vertical',
                        action='store_true')

    args = parser.parse_args()
    return args


def initFacialLandmarksEngine(args):
    global facialLandmarksEngineObj
    facialLandmarksEngineObj = FacialLandmarksEngine(args_cpu=args.cpu, args_debug=args.debug)

# main #############################################################################
def main():
    args = parse_args()

    init_glut()

    init_opengl() 

    glutTimerFunc(0, timer, 0)

    glutIdleFunc(Idle)

    initFacialLandmarksEngine(args)

    return glutMainLoop()

if __name__ == '__main__':
    sys.exit(main())

