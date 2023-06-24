import torch
import argparse

from nerf.gui import NeRFGUI
from nerf.network import NeRFNetwork
from nerf.utils import *
from nerf.opt import *

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    opt = get_opt()

    if opt.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif opt.data_format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    else: # 'nerf
        from nerf.provider import NeRFDataset
    
    # convert ratio to steps
    opt.refine_steps = [int(round(x * opt.iters)) for x in opt.refine_steps_ratio]

    seed_everything(opt.seed)

    model = NeRFNetwork(opt)
    
    criterion = torch.nn.MSELoss(reduction='none')
    # criterion = torch.nn.SmoothL1Loss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:
        
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            if not opt.test_no_video:
                test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

                if test_loader.has_gt:
                    trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)] # set up metrics
                    trainer.evaluate(test_loader) # blender has gt, so evaluate it.

                trainer.test(test_loader, write_video=True) # test and save video
            
            if not opt.test_no_mesh:
                if opt.stage == 1:
                    trainer.export_stage1(resolution=opt.texture_size)
                else:
                    # need train loader to get camera poses for visibility test
                    if opt.mesh_visibility_culling:
                        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
                    trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
        
    else:
        
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        save_interval = max(1, max_epoch // max(opt.n_ckpt, 1))
        eval_interval = max(1, max_epoch // max(opt.n_eval, 1))
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}.')

        if opt.ind_dim > 0:
            assert len(train_loader) < opt.ind_num, f"[ERROR] dataset too many frames: {len(train_loader)}, please increase --ind_num to at least this number!"

        # colmap can estimate a more compact AABB
        if opt.data_format == 'colmap':
            model.update_aabb(train_loader._data.pts_aabb)

        # scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.iters // 2, opt.iters * 3 // 4, opt.iters * 9 // 10], gamma=0.33)
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.01 + 0.99 * (iter / 500) if iter <= 500 else 0.1 ** ((iter - 500) / (opt.iters - 500)))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95 if opt.stage == 0 else None, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval, save_interval=save_interval)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()

            trainer.metrics = [PSNRMeter(),]
            trainer.train(train_loader, valid_loader, max_epoch)
            
            # last validation
            trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
            trainer.evaluate(valid_loader)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video
            
            if opt.stage == 1:
                trainer.export_stage1(resolution=opt.texture_size)
            else:
                trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
