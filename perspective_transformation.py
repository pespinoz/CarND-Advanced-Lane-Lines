import pickle
import matplotlib.pyplot as plt
from helper_functions import *

create_or_rewrite(TRANS_IMS)                       # create directory for choosing best transformation

########################################################################################################################

ctes = range(100, 190)
for cte in ctes:
    straight1 = cv2.imread(STRAIGHT_IMS[0])
    straight1 = cv2.cvtColor(straight1, cv2.COLOR_BGR2RGB)
    straight2 = cv2.imread(STRAIGHT_IMS[1])
    straight2 = cv2.cvtColor(straight2, cv2.COLOR_BGR2RGB)

    roi1 = [[-cte, IM_SIZE[1]], [int(IM_SIZE[0]/2-70), 446], [int(IM_SIZE[0]/2)+70, 446],
            [IM_SIZE[0]+cte, IM_SIZE[1]]]
    offsetx = 100
    offsety = 0
    roi2 = [[offsetx, IM_SIZE[1] - offsety], [offsetx, offsety], [IM_SIZE[0] - offsetx, offsety],
            [IM_SIZE[0] - offsetx, IM_SIZE[1] - offsety]]

    src = np.float32(roi1)
    dst = np.float32(roi2)

    M = cv2.getPerspectiveTransform(src, dst)  # use cv2.getPerspectiveTransform() to get M, the transform matrix
    warped1 = cv2.warpPerspective(straight1, M, IM_SIZE)  # use cv2.warpPerspective to get a top-down view
    warped2 = cv2.warpPerspective(straight2, M, IM_SIZE)  # use cv2.warpPerspective to get a top-down view

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=SCREENSIZE)
    draw_lines(straight1, [[roi1[0] + roi1[1]], [roi1[1] + roi1[2]], [roi1[2] + roi1[3]],
                           [roi1[3] + roi1[0]]], color=[255, 0, 0], thickness=2)
    ax1.imshow(straight1)
    ax1.set_title('Undistorted Image', fontsize=15)
    draw_lines(warped1, [[roi2[0] + roi2[1]], [roi2[1] + roi2[2]], [roi2[2] + roi2[3]],
                         [roi2[3] + roi2[0]]], color=[255, 0, 0], thickness=2)
    ax2.imshow(warped1)
    ax2.set_title('Warped Image', fontsize=15)
    draw_lines(straight2, [[roi1[0] + roi1[1]], [roi1[1] + roi1[2]], [roi1[2] + roi1[3]],
                           [roi1[3] + roi1[0]]], color=[255, 0, 0], thickness=2)
    ax3.imshow(straight2)
    ax3.set_title('Undistorted Image', fontsize=15)
    draw_lines(warped2, [[roi2[0] + roi2[1]], [roi2[1] + roi2[2]], [roi2[2] + roi2[3]],
                         [roi2[3] + roi2[0]]], color=[255, 0, 0], thickness=2)
    ax4.imshow(warped2)
    ax4.set_title('Warped Image', fontsize=15)
    fig.savefig(TRANS_IMS + 'n_' + str(cte) + '_straight_warped.jpg', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                pad_inches=0.1, frameon=None)
    plt.close(fig)

    ####################################################################################################################

# OK, I choose 160
# Save the transformation result for later use

cte = 160

straight1 = cv2.imread(STRAIGHT_IMS[0])
straight1 = cv2.cvtColor(straight1, cv2.COLOR_BGR2RGB)
straight2 = cv2.imread(STRAIGHT_IMS[1])
straight2 = cv2.cvtColor(straight2, cv2.COLOR_BGR2RGB)

roi1 = [[-cte, IM_SIZE[1]], [int(IM_SIZE[0] / 2 - 70), 446], [int(IM_SIZE[0] / 2) + 70, 446],
        [IM_SIZE[0] + cte, IM_SIZE[1]]]
offsetx = 100
offsety = 0
roi2 = [[offsetx, IM_SIZE[1] - offsety], [offsetx, offsety], [IM_SIZE[0] - offsetx, offsety],
        [IM_SIZE[0] - offsetx, IM_SIZE[1] - offsety]]

src = np.float32(roi1)
dst = np.float32(roi2)

M = cv2.getPerspectiveTransform(src, dst)  # use cv2.getPerspectiveTransform() to get M, the transform matrix
warped1 = cv2.warpPerspective(straight1, M, IM_SIZE)  # use cv2.warpPerspective to get a top-down view
warped2 = cv2.warpPerspective(straight2, M, IM_SIZE)  # use cv2.warpPerspective to get a top-down view

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=SCREENSIZE)
draw_lines(straight1, [[roi1[0] + roi1[1]], [roi1[1] + roi1[2]], [roi1[2] + roi1[3]],
                       [roi1[3] + roi1[0]]], color=[255, 0, 0], thickness=2)
ax1.imshow(straight1)
ax1.set_title('Undistorted Image', fontsize=15)
draw_lines(warped1, [[roi2[0] + roi2[1]], [roi2[1] + roi2[2]], [roi2[2] + roi2[3]],
                     [roi2[3] + roi2[0]]], color=[255, 0, 0], thickness=2)
ax2.imshow(warped1)
ax2.set_title('Warped Image', fontsize=15)
draw_lines(straight2, [[roi1[0] + roi1[1]], [roi1[1] + roi1[2]], [roi1[2] + roi1[3]],
                       [roi1[3] + roi1[0]]], color=[255, 0, 0], thickness=2)
ax3.imshow(straight2)
ax3.set_title('Undistorted Image', fontsize=15)
draw_lines(warped2, [[roi2[0] + roi2[1]], [roi2[1] + roi2[2]], [roi2[2] + roi2[3]],
                     [roi2[3] + roi2[0]]], color=[255, 0, 0], thickness=2)
ax4.imshow(warped2)
ax4.set_title('Warped Image', fontsize=15)
fig.savefig(TRANS_IMS + 'final_straight_warped.jpg', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
            pad_inches=0.1, frameon=None)

with open('pickle_files/transformation.pickle', 'wb') as f:
    pickle.dump(M, f)
