import os
import re
import cv2
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class RadarGhostDataset:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.metadata = self.generate_metadata()
        self.class2str = {
            -1:'background',
            1: 'pedestrian',
            2: 'cyclist',
            3: 'car',
            4: 'large_vehicle',
            5: 'motorcycle'
        }
    def generate_metadata(self) -> pd.DataFrame:
        """Generate metadata dataframe for the Radar Ghost dataset.

        This function generates a dataframe that contains metadata about each scenario in the dataset. The metadata includes
        the scenario number, sequence number, object class, split (train, val, or test), and the path to the corresponding
        h5 file.

        Returns:
            pd.DataFrame: metadata dataframe
        """
        metadata = []
        for split in ('train', 'val', 'test'):
            for h5 in self.data_root.glob(f'{split}/*'):
                m = re.match(rf"scenario-(\d+)_sequence-(\d+)_([^_]+)_{split}", h5.stem)
                scen, seq, class_ = m.groups()
                metadata.append(dict(
                    scenario=int(scen),
                    sequence=int(seq),
                    object_class=class_,
                    split = split,
                    path = h5.stem
                ))
        metadata_df = pd.DataFrame(metadata).sort_values(['scenario', 'sequence']).set_index(['scenario', 'sequence'])
        return metadata_df

    def get_scenario_image(self, idx: int):
        assert 0< idx < 22, "invalid scenario number! Choose an integer in range 0-21."
        img_file = self.data_root / 'images' / f'scenario-{idx:02d}.jpg'
        img = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)
        return img

    def get_lidar_pc(self, scenario: int, sequence: int):
        row = self.metadata.loc[scenario, sequence]
        path = os.path.join(self.data_root, row.split, row.path + ".h5")
        return pd.read_hdf(path, key='lidar')


    def get_radar_pc(self, scenario: int, sequence: int, sensor: str = 'right', raw: bool=False) -> pd.DataFrame:
        """Get radar point clouds for a given scenario and sequence.

        Args:
            scenario (int): scenario number
            sequence (int): sequence number
            sensor (str, optional): radar sensor. Defaults to 'right'.
            raw (bool, optional): if True, returns the raw radar point cloud.
                Otherwise, applies some processing to the data. Defaults to False.

        Returns:
            pd.DataFrame: radar point cloud data
        """
        row = self.metadata.loc[scenario, sequence]
        path = os.path.join(self.data_root, row.split, row.path + ".h5")

        radar_df = pd.read_hdf(path, key='radar')
        if sensor in ['left', 'right']:
            radar_df = radar_df.query(f"sensor=='{sensor}'").copy()

        if raw:
            return radar_df
        else:
            label_id = radar_df.label_id.copy().values

            background = label_id == 0
            ignore = label_id == -1
            noise = label_id == -2
            ignore = np.logical_or(ignore, noise)  # treat noise as ignore

            # differentiate between special labels (one digit) and 4 digit labels
            bg_or_ign = np.logical_or(background, ignore)
            non_bg_or_ign = np.logical_not(bg_or_ign)

            sketchy = label_id < -2

            label_id = np.abs(label_id)

            c = (label_id // 1000) % 10
            m = (label_id // 100) % 10
            t = (label_id // 10) % 10
            o = label_id % 10

            # c, m, t, o are only valid for non background/ignore/noise detections

            # set to -1 for background/ignore/noise -> maybe useful for later
            c[bg_or_ign] = -1
            m[bg_or_ign] = -1
            t[bg_or_ign] = -1
            o[bg_or_ign] = -1

            radar_df['is_background'] = bg_or_ign
            radar_df['class'] = [self.class2str[x] for x in c]
            radar_df['class_id'] = c
            radar_df['is_main_obj'] = m
            radar_df['multipath_type'] = t
            radar_df['multipath_order'] = o

            radar_df['phi_deg'] = np.rad2deg(radar_df.phi_sc)
            radar_df['amp_log'] = np.log(radar_df.amp)
            radar_df.human_readable_label = radar_df.human_readable_label.astype('category')
            radar_df.set_index('frame', inplace=True)
            return radar_df
            # return self.process_radar(radar_df)


    # def process_radar(self, df) -> pd.DataFrame:
    #     """Convert radar pc columns for easier processing.
    #     code is based on the suggested parsing from the dataset's github repo:
    #      https://github.com/flkraus/ghosts/blob/main/label_convention.md
    #
    #     :param df: raw radar pc
    #     :return: processed radar pc
    #     """
    #     label_id = df.label_id.copy().values
    #
    #     background = label_id == 0
    #     ignore = label_id == -1
    #     noise = label_id == -2
    #     ignore = np.logical_or(ignore, noise)  # treat noise as ignore
    #
    #     # differentiate between special labels (one digit) and 4 digit labels
    #     bg_or_ign = np.logical_or(background, ignore)
    #     non_bg_or_ign = np.logical_not(bg_or_ign)
    #
    #     sketchy = label_id < -2
    #
    #     label_id = np.abs(label_id)
    #
    #     c = (label_id // 1000) % 10
    #     m = (label_id // 100) % 10
    #     t = (label_id // 10) % 10
    #     o = label_id % 10
    #
    #     # c, m, t, o are only valid for non background/ignore/noise detections
    #
    #     # set to -1 for background/ignore/noise -> maybe useful for later
    #     c[bg_or_ign] = -1
    #     m[bg_or_ign] = -1
    #     t[bg_or_ign] = -1
    #     o[bg_or_ign] = -1
    #
    #     df['is_background'] = bg_or_ign
    #     df['class'] = [self.class2str[x] for x in c]
    #     df['class_id'] = c
    #     df['is_main_obj'] = m
    #     df['multipath_type'] = t
    #     df['multipath_order'] = o
    #
    #     df['phi_deg'] = np.rad2deg(df.phi_sc)
    #     df.human_readable_label = df.human_readable_label.astype('category')
    #     return df

    
def get_clusters(radar_df, idx, agg=0, debug=False):
    """Get radar clusters for a given index.

    Args:
        radar_df (pd.DataFrame): radar point cloud data
        idx (int): index of the point cloud
        agg (int, optional): number of frame to aggregate.
            Defaults to 0, which means no aggregation.

    Returns:
        tuple: tuple containing 3 pandas.DataFrames:
            - candidate real target (np.array (y,x)  )
            - candidate ghost target (np.array (y,x) )
            - static points (pd.DataFrame)
    """
    frame = radar_df.loc[idx:idx+agg].reset_index()

    stat = frame.query('instance_id == 0')
    dyn = frame.query('instance_id > 0').query('abs(vr_sc)>0.2')
    if len(dyn) == 0 or 2 not in dyn.multipath_type.unique() or 2 not in dyn.multipath_order.unique():
        return None, None, None
    if debug:
        main_obj_id = dyn.query('multipath_order==1').instance_id.values[0]
    else:
        main_obj_id = dyn.instance_id.min()
    clst0_df = dyn[dyn.instance_id == main_obj_id]

    # dyn_filt = dyn[dyn.multipath_type == 2]
    dyn_filt = dyn[(dyn.multipath_type == 2) & (dyn.multipath_order == 2)]
    if len(dyn_filt) == 0:
        return None, None, None
    clst1_df = dyn[dyn.instance_id == dyn_filt.instance_id.value_counts().idxmax()]
    if len(clst1_df) < 2:
        return None, None, None

    clst0 = clst0_df[['x_cc', 'y_cc']].values
    clst1 = clst1_df[['x_cc', 'y_cc']].values
    return clst0, clst1, stat


def estimate_reflector_points(clst0, clst1, debug=False):
    """implementation of multipath mitigation method from https://ieeexplore.ieee.org/abstract/document/9455253

    :param clst0:
    :param clst1:
    :param debug:
    :return:
    """
    r = []
    A = []
    B = []
    Beta = []
    epsilon = 1e-8  # to avoid zero division issues
    for p0, p1 in zip(clst0, clst1):
        x0, y0 = p0
        x1, y1 = p1
        # y0, x0 = p0
        # y1, x1 = p1
        m = (y1-y0)/(x1-x0 + epsilon) # reflection surface slope
        a = np.arctan(m) - np.pi/2 # reflection surface angle
        b = (y0+y1)/2-np.tan(a)*(x0+x1)/2 # reflection surface offset
        beta = np.arctan(y1/x1) # ghost line angle


        # intersection
        x_r = b / (np.tan(beta) - np.tan(a))
        y_r = np.tan(beta) * x_r
        r.append((x_r,y_r))

        if debug:
            A.append(a)
            B.append(b)
            Beta.append(beta)

    if debug:
        return np.array(r), np.array(A), np.array(B), np.array(Beta)
    else:
        return np.array(r)



def plot_results(radar_pc, algo_res, linreg_res, debug=False, xlim=None, ylim=None, title=""):
    """

    :param radar_pc: list/tuple containing 2 radar clusters (real-ghost pair) and the static points
    :param algo_res: results of the multipath mitigation algorithm
    :param linreg_res: results of linear regression on the estimated reflector points
    :param debug: if True returns all parameters. If False returns only estimated reflector points.
    :param xlim: figure limits
    :param ylim: figure limits
    :param title: figure title
    :return: plt figure visualizing the results
    """
    clst0, clst1, stat = radar_pc
    if debug:
        r, a, b, beta = algo_res

    else:
        r = algo_res
    x = np.arange(r[:,0].min()-1,r[:,0].max()+1)


    fig, ax = plt.subplots(figsize=(9,6))
    plt.scatter(x=0, y=0, c='k', s=100)
    stat.plot.scatter(x='x_cc', y='y_cc', c='green', s=1, ax=ax, label="Static Points")
    plt.scatter(clst0[:,0], clst0[:,1], c='blue', s=10, label="Real Target Candidate")
    plt.scatter(clst1[:,0], clst1[:,1], c='magenta', s=10, label="Ghost Target Candidate")
    plt.scatter(x=r[:,0], y=r[:,1], c='black', s=10, label="Estimated Reflection Surface")

    # plot reflector lien
    if debug and False:
        reflector_line = np.tan(linreg_res.beta.mean()) * x
        plt.plot(x, reflector_line, label="Reflector Line")
    plt.plot(x, linreg_res.intercept + linreg_res.slope*x, 'r', label='Fitted Reflection Surface')

    if debug:
        x_ghost = np.arange(0, clst1[:,0].mean())
        ghost_line = np.tan(beta.mean()) * x_ghost
        plt.plot(x_ghost, ghost_line, linestyle='--', label='Ghost Line')



    ylim = ylim or [-30, 10]
    xlim = xlim or [0, 40]
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    # ax.invert_xaxis()
    # plt.legend(loc='upper left')
    plt.legend(loc='lower left')
    plt.title(title + f"clst0: {clst0.shape[0]} pts | clst1: {clst1.shape[0]} pts")
    return fig

#################
#    Metrics    #
#################

def MSD(r, m, b):
    x = r[:, 0]
    y = r[:, 1]
    d = abs((y - m * x - b)) / (np.sqrt(m ** 2 + 1))
    return d.mean()


def PER(r, m, b):
    pass


def ANG(r, a):
    a_hat = []
    for x, y in r:
        a_hat.append(np.arctan(y / x))

    return np.rad2deg((np.array(a_hat) - a).mean())