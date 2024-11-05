import numpy as np
import OpenGL.GL as gl
import pangolin
import cv2
from OpenGL.GLUT import *
from OpenGL.GL import *
from multiprocessing import Queue, Process



class Viewer(object):
    def __init__(self):
        self.image_queue = Queue()
        self.pose_queue = Queue()
        self.uwb_queue = Queue()
        self.feature_point_queue = Queue()

        self.view_thread = Process(target=self.view)
        self.view_thread.start()
        self.stopped = False

    def update_pose(self, pose):
        if pose is None:
            return
        self.pose_queue.put(pose.matrix())

    def update_anchor(self, anchor_pos):
        if anchor_pos is None:
            return
        self.uwb_queue.put(anchor_pos.uwb_positions)

    def update_image(self, image):
        if image is None:
            return
        elif image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        self.image_queue.put(image)

    def update_feature_points(self, feature_points):
        if feature_points is None:
            return
        self.feature_point_queue.put(feature_points)

    def draw_text(self, text, x, y):
        """
        Draw text using GLUT at the given window coordinates.
        """
        glWindowPos2f(x, y)  # Position in screen space
        for ch in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

    def view(self):
        glutInit()
        pangolin.CreateWindowAndBind('Viewer', 1200, 768)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        viewpoint_x = 0
        viewpoint_y = -7  
        viewpoint_z = -18 
        viewpoint_f = 1000

        proj = pangolin.ProjectionMatrix(
            1024, 768, viewpoint_f, viewpoint_f, 512, 389, 0.1, 300)
        look_view = pangolin.ModelViewLookAt(
            viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)

        # Camera Render Object (for view / scene browsing)
        scam = pangolin.OpenGlRenderState(proj, look_view)

        # Add named OpenGL viewport to window and provide 3D Handler
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 175 / 1024., 1.0, -1024 / 768.)
        dcam.SetHandler(pangolin.Handler3D(scam))

        # image
        width, height = 376, 240
        dimg = pangolin.Display('image')
        dimg.SetBounds(0, height / 768., 0.0, width / 1024., 1024 / 768.)
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.ones((height, width, 3), 'uint8')

        # axis
        axis = pangolin.Renderable()
        axis.Add(pangolin.Axis())


        trajectory = DynamicArray()
        camera = None
        image = None

        while not pangolin.ShouldQuit():

            if not self.uwb_queue.empty():
                while not self.uwb_queue.empty():
                    uwb_positions = self.uwb_queue.get()
                self.uwb_anchors = uwb_positions

            if not self.pose_queue.empty():
                while not self.pose_queue.empty():
                    pose = self.pose_queue.get()
                trajectory.append(pose[:3, 3])
                camera = pose

            if not self.image_queue.empty():
                while not self.image_queue.empty():
                    img = self.image_queue.get()
                img = img[::-1, :, ::-1]
                img = cv2.resize(img, (width, height))
                image = img.copy()


            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)


            # draw axis
            axis.Render()

            # draw current camera
            if camera is not None:
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawCameras(np.array([camera]), 0.5)

                # Display UWB anchor coordinates at the bottom right corner as a vector
                for i, anchor in enumerate(self.uwb_anchors):
                    # Set the x and y screen coordinates for the text
                    y_position = 140 - i * 20  # Adjust Y position for each anchor
                    anchor_text = f"Anchor {i + 1}: ({anchor[0]:.2f}, {anchor[1]:.2f}, {anchor[2]:.2f})"
                    self.draw_text(anchor_text, 850, y_position)  # Adjust x, y as needed for screen

                # Get the coordinates of the robot pose
                pose_position = camera[:3, 3]
                # Format the text as "Robot position: [X, Y, Z]"
                coord_text = f"Robot position: [{pose_position[0]:.2f}, {pose_position[1]:.2f}, {pose_position[2]:.2f}]"

                # Calculate position for bottom right corner (10 pixels padding from right and bottom)
                text_x = 850  # Rough approximation based on text length
                text_y = 20  # 10 pixels from the bottom of the screen

                # Drawing text on the screen in the bottom right corner
                self.draw_text(coord_text, text_x, text_y)

            # show trajectory
            if len(trajectory) > 0:
                gl.glPointSize(2)
                gl.glColor3f(0.0, 0.0, 0.0)
                pangolin.DrawPoints(trajectory.array())

            # show image
            if image is not None:
                texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()
                
            pangolin.FinishFrame()


class DynamicArray(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)

        self.data = np.zeros((1000, *shape))
        self.shape = shape
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])
    
    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (2 * len(self.data), *self.shape) , refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind+len(xs)] = xs
        else:
            for i, x in enumerate(xs):
                self.data[self.ind+i] = x
            self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[:self.ind]:
            yield x




if __name__ == '__main__':
    import g2o
    import time

    viewer = Viewer()
    viewer.update_pose(g2o.Isometry3d())