from ..util.argparse import FsExistsType


class Filter(object):
    @classmethod
    def add_to_parser(cls, subparsers):
        filter_group = subparsers.add_parser("filter")
        filter_group.add_argument("in_json", type=FsExistsType())
        filter_group.add_argument("out_dir", type=FsExistsType())
        filter_group.add_argument("--ds", default=15, type=int, help="Disk size")
        filter_group.set_defaults(starfish_command=Filter.run)

    @classmethod
    def run(cls, args):
        import numpy as np

        from ..filters import white_top_hat
        from ..io import Stack

        print('Filtering ...')
        print('Reading data')
        s = Stack()
        s.read(args.in_json)

        # filter raw images, for all hybs and channels
        stack_filt = []
        for im_num, im in enumerate(s.squeeze()):
            print("Filtering image: {}...".format(im_num))
            im_filt = white_top_hat(im, args.ds)
            stack_filt.append(im_filt)

        stack_filt = s.un_squeeze(stack_filt)

        # filter dots
        print("Filtering dots ...")
        dots_filt = white_top_hat(s.aux_dict['dots'], args.ds)

        print("Writing results ...")
        # create a 'stain' for segmentation
        stain = np.mean(s.max_proj('ch'), axis=0)
        stain = stain / stain.max()

        # update stack
        s.set_stack(stack_filt)
        s.set_aux('dots', dots_filt)
        s.set_aux('stain', stain)

        s.write(args.out_dir)
