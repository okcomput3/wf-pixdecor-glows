#ifndef PIXDECOR_3D_EFFECT_HPP
#define PIXDECOR_3D_EFFECT_HPP

#include <wayfire/view-transform.hpp>
#include <wayfire/opengl.hpp>
#include <wayfire/scene-render.hpp>
#include <wayfire/scene.hpp>
#include <wayfire/toplevel-view.hpp>
#include <wayfire/view.hpp>
#include <wayfire/nonstd/wlroots.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec4.hpp>
#include <memory>

namespace wf
{
namespace pixdecor
{

}  // namespace pixdecor
}  // namespace wf

namespace wf
{
class pixdecor_3d_effect_t : public wf::scene::view_2d_transformer_t
{
private:
    wayfire_toplevel_view _view;
    OpenGL::program_t shader_program;
    bool shader_initialized = false;

    const char* vertex_shader = R"(
#version 100
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
uniform mat4 mvp;

void main() {
    v_texcoord = texcoord;
    gl_Position = mvp * vec4(position, 0.0, 1.0);
}
)";

    const char* fragment_shader = R"(
#version 100
precision mediump float;
varying vec2 v_texcoord;
uniform vec2 resolution;
uniform float time;
uniform vec4 color;

// Plasma effect shader
void main() {
    vec2 uv = v_texcoord;
    vec2 p = (uv * 2.0 - 1.0) * vec2(resolution.x / resolution.y, 1.0);
    
    float t = time * 0.5;
    
    // Create plasma waves
    float wave1 = sin(p.x * 10.0 + t);
    float wave2 = sin(p.y * 10.0 + t * 1.3);
    float wave3 = sin((p.x + p.y) * 8.0 + t * 0.8);
    float wave4 = sin(length(p) * 12.0 - t * 2.0);
    
    float plasma = (wave1 + wave2 + wave3 + wave4) * 0.25;
    
    // Color gradient
    vec3 col1 = vec3(0.5, 0.0, 1.0); // Purple
    vec3 col2 = vec3(0.0, 0.5, 1.0); // Cyan
    vec3 col3 = vec3(1.0, 0.3, 0.5); // Pink
    
    vec3 finalColor = mix(col1, col2, sin(plasma * 3.14159 + t) * 0.5 + 0.5);
    finalColor = mix(finalColor, col3, cos(plasma * 2.0 - t * 0.5) * 0.5 + 0.5);
    
    // Add some glow
    float glow = 1.0 - length(p) * 0.5;
    finalColor += vec3(0.2, 0.1, 0.3) * glow;
    
    gl_FragColor = vec4(finalColor * color.rgb, color.a * 0.7);
}
)";

    void init_shader()
    {
        if (shader_initialized) return;
        
        shader_program.set_simple(
            OpenGL::compile_program(vertex_shader, fragment_shader));
        shader_initialized = true;
    }

public:
    pixdecor_3d_effect_t(wayfire_toplevel_view view)
        : view_2d_transformer_t(view), _view(view)
    {}

    ~pixdecor_3d_effect_t() 
    {
        shader_program.free_resources();
    }

    void gen_render_instances(std::vector<wf::scene::render_instance_uptr>& instances,
        wf::scene::damage_callback push_damage, wf::output_t *output = nullptr) override
    {
        // First add our custom render instance for the shader back layer (renders behind)
        instances.push_back(std::make_unique<effect_render_instance_t>(this, push_damage));
        
        // Then add the parent's render instances (the actual view at original size, renders on top)
        view_2d_transformer_t::gen_render_instances(instances, push_damage, output);
    }

    wf::geometry_t get_bounding_box() override
    {
        if (!_view) return {0, 0, 0, 0};
        auto geo = _view->get_geometry();
        
        // Calculate the maximum offset from the layers behind
        int max_layer = 1;
        float max_scale = 1.0f + (max_layer * -0.2f); // Layers grow bigger behind
        float max_dx = max_layer * 1.0f;
        float max_dy = max_layer * 1.0f;
        
        // Calculate how much bigger the largest layer is
        int extra_width = int(geo.width * (max_scale - 1.0f) / 2.0f);
        int extra_height = int(geo.height * (max_scale - 1.0f) / 2.0f);
        
        // Account for both the scaling and the offset
        int offset_x = std::max(int(max_dx + extra_width), 0);
        int offset_y = std::max(int(max_dy + extra_height), 0);
        
        return wf::geometry_t{
            geo.x - offset_x,
            geo.y - offset_y,
            geo.width + 2 * offset_x,
            geo.height + 2 * offset_y
        };
    }

    class effect_render_instance_t : public wf::scene::render_instance_t
    {
        pixdecor_3d_effect_t *self;
        wf::scene::damage_callback push_damage;
        float animation_time = 0.0f;

    public:
        effect_render_instance_t(pixdecor_3d_effect_t *s, wf::scene::damage_callback d)
            : self(s), push_damage(d)
        {}

        void schedule_instructions(std::vector<wf::scene::render_instruction_t>& instructions,
            const wf::render_target_t& target, wf::region_t& damage) override
        {
            auto our_region = self->get_bounding_box();
            wf::region_t our_damage = damage & our_region;
            if (!our_damage.empty())
            {
                instructions.push_back(wf::scene::render_instruction_t{
                    .instance = this,
                    .target = target,
                    .damage = std::move(our_damage),
                });
            }
        }

        void render(const wf::scene::render_instruction_t& data) override
        {
            if (!self->_view || !self->_view->is_mapped()) return;

            self->init_shader();
            
            auto view_geo = self->_view->get_geometry();

            wf::gles::bind_render_buffer(data.target);

            GL_CALL(glEnable(GL_BLEND));
            GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

            animation_time += 0.016f; // Approximately 60 FPS

            // Get framebuffer dimensions from the target's scale box
            int fb_width = data.target.geometry.width;
            int fb_height = data.target.geometry.height;
            
            // Get projection matrix
            auto ortho = glm::ortho(0.0f, (float)fb_width,
                                   (float)fb_height, 0.0f);

            // Render shader effect layers BEHIND the window (from back to front)
            for (int i = 1; i > 0; i--)
            {
                // Layers get progressively bigger as they go back
                float scale = 1.0f + (i * 0.2f); // Each layer 2% bigger
                float dx = i * 1.0f;  // Offset by i pixels horizontally
                float dy = i * 1.0f;  // Offset by i pixels vertically
                
                // Center the scaled layer
                float cx = (view_geo.width * (scale - 1.0f)) / 2.0f;
                float cy = (view_geo.height * (scale - 1.0f)) / 2.0f;
                
                wf::geometry_t layer_geo{
                    int(view_geo.x + dx - cx),
                    int(view_geo.y + dy - cy),
                    int(view_geo.width * scale),
                    int(view_geo.height * scale)
                };

                self->shader_program.use(wf::TEXTURE_TYPE_RGBA);
                self->shader_program.uniformMatrix4f("mvp", ortho);
                self->shader_program.uniform2f("resolution", layer_geo.width, layer_geo.height);
                self->shader_program.uniform1f("time", animation_time + i * 0.05f);
                
                // Fade out layers that are further back
                float alpha = 0.05f + (1.0f - i / 20.0f) * 0.2f;
                glm::vec4 layer_color(1.0f, 1.0f, 1.0f, alpha);
                self->shader_program.uniform4f("color", layer_color);

                wf::region_t layer_reg(layer_geo);
                auto layer_damage = data.damage & layer_reg;
                
                for (const auto& box : layer_damage)
                {
                    wf::gles::render_target_logic_scissor(data.target, wlr_box_from_pixman_box(box));
                    
                    GLfloat x = layer_geo.x;
                    GLfloat y = layer_geo.y;
                    GLfloat w = layer_geo.width;
                    GLfloat h = layer_geo.height;
                    
                    GLfloat vertexData[] = {
                        x, y,
                        x + w, y,
                        x + w, y + h,
                        x, y + h
                    };
                    
                    GLfloat texData[] = {
                        0.0f, 0.0f,
                        1.0f, 0.0f,
                        1.0f, 1.0f,
                        0.0f, 1.0f
                    };

                    self->shader_program.attrib_pointer("position", 2, 0, vertexData);
                    self->shader_program.attrib_pointer("texcoord", 2, 0, texData);
                    
                    GL_CALL(glDrawArrays(GL_TRIANGLE_FAN, 0, 4));
                }
                
                self->shader_program.deactivate();
            }

            GL_CALL(glDisable(GL_BLEND));
            
            // Schedule next frame for animation
            push_damage(self->get_bounding_box());
        }
    };
};
}  // namespace wf

#endif  // PIXDECOR_3D_EFFECT_HPP