import os
import pandas as pd
from app import app
from flask import render_template, flash, redirect, url_for, request, send_from_directory
from werkzeug.urls import url_parse
from werkzeug.utils import secure_filename
import pickle

if not app.config['NO_MODEL']:
    from fpm import *
    from params import *


def boolean(s):
    """Internal function"""
    return True if s=='on' else False

@app.route('/')
@app.route('/index')
def index():
    files = [file for file in os.listdir(app.config['UPLOAD_FOLDER'])
             if file.endswith('.tiff') or file.endswith('.tif')]
    models = app.config['MODELS_LIST']

    return render_template(
        'index.html',
        title='Home',
        files=files,
        models=models,
        heading="About",
        subtitle='Illumination angle estimation using neural networks. '
        'Try it out!'
        )


@app.route('/upload')
def upload_file():
    return render_template('upload.html', title='Upload the FPM files')


@app.route('/uploader', methods=['POST'])
def upload_file_():
    cause = ''  # For catching specific cause of error
    if request.method == 'POST':
        _, format = os.path.splitext(request.files['f'].filename)
        name = request.form['name']

        try:
            cause = 'while uploading the files. Ensure that the files'
            ' are accessible and try again. '

            f = request.files['f']
            f.save(
             os.path.join(
               app.config['UPLOAD_FOLDER'],
               secure_filename(
                    f"{name.replace('.tiff', '').replace('.tif', '')}"
                    f"{format}")
               )
            )

            cause = 'while reading the parameters. Make sure that you have'
            'not entered any non-numerical characters in any of the fields.'

            NA = float(request.form['NA'])
            PIXELSIZE = float(request.form['PIXELSIZE'])
            RI = float(request.form['RI'])
            WAVELENGTH = float(request.form['WAVELENGTH'])
            IMAGESIZE = float(request.form['IMAGESIZE'])
            MAGNIFICATION = float(request.form['MAGNIFICATION'])

            params = pd.DataFrame(  # Metadata
                {  # DO NOT remove square brackets. They are needed for correct shape.
                    'NA': [NA],
                    'PIXELSIZE': [PIXELSIZE],  # um
                    'RI': [RI],
                    'WAVELENGTH': [WAVELENGTH],  # um
                    'IMAGESIZE': [IMAGESIZE],  # In case of non-square images, put lesser of the two dimensions here
                    'MAGNIFICATION': [MAGNIFICATION],

                     # Not used for real-life data. Left here for compatibility with old code. Does not need to be changed
                    'ILLUMINATION_OFFCENTER_X': [0],
                    'ILLUMINATION_OFFCENTER_Y': [0],
                    'FRAMES': [0]
                }
            )

            cause = 'while saving the parameters\' file. Make sure that the'
            'disk is not write-protected.'

            pickle.dump(
                params,
                open(os.path.join(
                        app.config['PARAMS_FOLDER'],
                        secure_filename(
                            f"{name.replace('.tiff', '').replace('.tif', '')}"
                            f"{format}.pkl"
                         )), 'wb')
            )

            flash(
                f"{name.replace('.tiff', '').replace('.tif', '')}"
                f"{format} was uploaded succesfully."
                )

            cause = None

        except Exception as e:
            flash(
                f'An error occured {cause}' if cause is not None else
                'An unknown error occured.')
            return f"""<div class="w3-container">
              <h1 class="w3-xxxlarge w3-text-black"><b>Sorry Something Went Wrong.</b></h1>
              <hr style="width:50px;border:5px solid red" class="w3-round">
              <p>An error occured while uploading the file. See below for more info.</p>
              <br />
              <h3 class="w3-xlarge w3-text-black"><b>Error Text:</b></h3>
              <hr>
              <p> {repr(e)} </p>
              <a href='/upload'><h3 class="w3-xlarge w3-text-black">
                <b>&lt; Go back and try again.</b></h3></a>
            </div>"""
        return redirect(url_for('index'))
    return redirect(url_for('index'))


@app.route('/analyze', methods=['POST'])
def analyzer():
    if request.method == 'POST':
        file = request.form['files_dropdown']
        print(file)
        model = request.form['models_dropdown']
        print(model)

#         try:
        app.config['PARAMS'] = pickle.load(open(os.path.join(
                app.config['PARAMS_FOLDER'],
                secure_filename(f"{file}.pkl")), 'rb'
                ))

        app.config['WORKING_FILE'] = file

        if app.config['NO_MODEL']:  # DEBUG
            app.config['LOADED_MODEL'] = 'DUMMY'
            app.config['MODEL'] = 'DUMMY'

        elif app.config['LOADED_MODEL'] is not None and \
            app.config['LOADED_MODEL'] == model and \
                app.config['MODEL'] is not None:
            pass
            # flash(f'{model} model is already loaded. Reusing...')

        else:
            cfg_path, preprocess_fft = app.config['MODELS'][model]
            cfg = pickle.load(open(os.path.join(cfg_path, 'cfg.pkl'), 'rb'))
            cfg.MODEL.WEIGHTS = os.path.join(cfg_path, 'model_final.pth')
            app.config['MODEL'] = Predictor(cfg, preprocess_fft=preprocess_fft)
            app.config['LOADED_MODEL'] = model
            # flash(f'Succesfully loaded the {model} model.')

#         except Exception as e:
#             flash(f'An error occured.')

#             return f"""<div class="w3-container">
#               <h1 class="w3-xxxlarge w3-text-black"><b>Sorry Something Went Wrong.</b></h1>
#               <hr style="width:50px;border:5px solid red" class="w3-round">
#               <p>An error occured while loading the model. See below for more info.</p>
#               <br />
#               <h3 class="w3-xlarge w3-text-black"><b>Error Text:</b></h3>
#               <hr>
#               <p> {repr(e)} </p>
#               <a href='/index'><h3 class="w3-xlarge w3-text-black">
#                 <b>&lt; Go back and try again.</b></h3></a>
#             </div>"""
        # return redirect(url_for('analyze'))

        return render_template(
            'analyze.html',
            title='Analysis UI',
            model=app.config['LOADED_MODEL'],
            file=app.config['WORKING_FILE']
        )


# @app.route('/analyze')
# def analyze():
#     if app.config['NO_MODEL']:  # DEBUG
#         # flash('Testing analyze page. Model loading is not required.')
#         app.config['LOADED_MODEL'] = 'DUMMY'
#         app.config['WORKING_FILE'] = 'DUMMY'
#
#     elif app.config['MODEL'] is None:
#         flash('No model loaded. Redirected back to the homepage.')
#         return redirect(url_for('index'))
#
#     return render_template(
#         'analyze.html',
#         title='Analysis UI',
#         model=app.config['LOADED_MODEL'],
#         file=app.config['WORKING_FILE']
#     )


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':

        # Get the data from the POST request and then
#         try:
        app.config['ILL_PARAMS']['predictor'] = app.config['MODEL']
        app.config['ILL_PARAMS']['tiff_path'] = os.path.join(
            app.config['UPLOAD_FOLDER'],
            app.config['WORKING_FILE']
        )

        # print(list(request.form.keys()))  # Debug

        # Illumination Params
        app.config['ILL_PARAMS']['window'] = str(request.form['window'])
        app.config['ILL_PARAMS']['a'] = float(request.form['a'])
        app.config['ILL_PARAMS']['p'] = float(request.form['p'])
        app.config['ILL_PARAMS']['sig'] = float(request.form['sig'])
        app.config['ILL_PARAMS']['do_psd'] = boolean(request.form.get('do_psd', 'off'))
        app.config['ILL_PARAMS']['starting_angle'] = float(request.form['starting_angle'])
        app.config['ILL_PARAMS']['increase_angle'] = boolean(request.form.get('increase_angle', 'off'))
        app.config['ILL_PARAMS']['calibrate'] = boolean(request.form.get('calibrate', 'off'))
        app.config['ILL_PARAMS']['fill_empty'] = boolean(request.form.get('fill_empty', 'off'))

        # Reconstruction Params
        app.config['REC_PARAMS']['scale'] = float(request.form['scale'])
        app.config['REC_PARAMS']['window'] = str(request.form['r_window'])
        app.config['REC_PARAMS']['a'] = float(request.form['r_a'])
        app.config['REC_PARAMS']['p'] = float(request.form['r_p'])
        app.config['REC_PARAMS']['sig'] = float(request.form['r_sig'])
        app.config['REC_PARAMS']['do_psd'] = boolean(request.form.get('r_do_psd', 'off'))
        app.config['REC_PARAMS']['n_iters'] = int(request.form['n_iters'])
        app.config['REC_PARAMS']['adaptive_noise'] = float(request.form['adaptive_noise'])
        app.config['REC_PARAMS']['adaptive_pupil'] = float(request.form['adaptive_pupil'])
        app.config['REC_PARAMS']['adaptive_img'] = float(request.form['adaptive_img'])
        app.config['REC_PARAMS']['alpha'] = float(request.form['alpha'])
        app.config['REC_PARAMS']['delta_img'] = float(request.form['delta_img'])
        app.config['REC_PARAMS']['delta_pupil'] = float(request.form['delta_pupil'])

        tiff_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            app.config['WORKING_FILE']
        )

        # Illumination Estimation
        discs, radii = get_illumination(**app.config['ILL_PARAMS'])

        # Save Illumination Estimation Results
        discs_path = save_illumination(
            discs,
            radii,
            tiff_path,
            app.config['SAVE_PARAMS']
        )

        # Reconstruction
        obj, pupil, imgs = get_reconstruction(
            tiff_path,
            discs,
            app.config['PARAMS'],
            app.config['REC_PARAMS']
        )

        # Save Reconstruction Results
        amp, phase, mean, p_amp, p_phase = save_reconstruction(
            obj,
            pupil,
            imgs,
            tiff_path,
            app.config['SAVE_PARAMS']
        )

        discs_path = f'/illumination-results/{os.path.basename(discs_path)}'
        amp = f'/reconstruction-results/{os.path.basename(amp)}'
        phase = f'/reconstruction-results/{os.path.basename(phase)}'
        mean = f'/reconstruction-results/{os.path.basename(mean)}'
        p_amp = f'/reconstruction-results/{os.path.basename(p_amp)}'
        p_phase = f'/reconstruction-results/{os.path.basename(p_phase)}'

#         except Exception as e:
#             flash(f'An error occured.')

#             return f"""<div class="w3-container">
#               <h1 class="w3-xxxlarge w3-text-black"><b>Sorry Something Went Wrong.</b></h1>
#               <hr style="width:50px;border:5px solid red" class="w3-round">
#               <p>An error occured while running the inference. See below for more info.</p>
#               <br />
#               <h3 class="w3-xlarge w3-text-black"><b>Error Text:</b></h3>
#               <hr>
#               <p> {repr(e)} </p>
#               <a href='/index'><h3 class="w3-xlarge w3-text-black">
#                 <b>&lt; Go back and try again.</b></h3></a>
#             </div>"""

        return render_template(
            'results.html',
            title='Results',
            heading=app.config['WORKING_FILE'],
            subtitle=app.config['LOADED_MODEL'],
            discs_path=discs_path,
            amp=amp,
            phase=phase,
            mean=mean,
            p_amp=p_amp,
            p_phase=p_phase,
            )


@app.route('/illumination-results/<path:path>')
def ill_output(path):
#     path1 = os.path.basename(app.config['OUTPUT_FOLDER'])
#     path2 = os.path.basename(app.config['SAVE_PARAMS']['illumination']['savedir'])
#     return 
    return send_from_directory(
        os.path.abspath(app.config['SAVE_PARAMS']['illumination']['savedir']),
        path
        )


@app.route('/reconstruction-results/<path:path>')
def rec_output(path):
    return send_from_directory(
        os.path.abspath(app.config['SAVE_PARAMS']['reconstruction']['savedir']),
        path
        )
# @app.route('/report', methods=['GET', 'POST'])
# def report():
#     if request.method == 'POST':
#
#         # Get the data from the POST request and then
#         try:
#             # something here
#             success = True
#             error = None
#         except Exception as e:
#             error = str(e)
#
#         return render_template(
#             'report.html',
#             title='Report',
#             heading='Lorem Ipsum',
#             subtitle='Dolor Sit amet',
#             success=success,
#             error=error
#             )
#     elif request.method == 'GET':
#         try:
#             # something here
#             success = True
#             error = None
#         except Exception as e:
#             error = str(e)
#
#         return render_template(
#             'report.html',
#             title='Report',
#             heading='Lorem Ipsum',
#             subtitle='Dolor Sit amet',
#             success=success,
#             error=error
#             )
