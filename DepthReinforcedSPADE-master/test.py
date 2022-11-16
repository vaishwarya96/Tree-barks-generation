"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.dual_model import DualModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = DualModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated_surface, generated_color = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated_surface.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_surface', generated_surface[b]),
                               ('synthesized_color', generated_color[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1], data_i['input_min'][b], data_i['input_max'][b])

webpage.save()
