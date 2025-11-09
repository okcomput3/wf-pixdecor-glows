#ifndef PIXDECOR_3D_SHADER_HPP
#define PIXDECOR_3D_SHADER_HPP

#include <wayfire/opengl.hpp>
#include <wayfire/scene-render.hpp>
#include <wayfire/scene.hpp>
#include <wayfire/toplevel-view.hpp>
#include <wayfire/view.hpp>
#include <wayfire/nonstd/wlroots.hpp>
#include <glm/vec4.hpp>
#include <memory>

namespace wf
{
class pixdecor_shader_node_t : public wf::scene::node_t
{
private:
    wayfire_toplevel_view _view;

public:
    pixdecor_shader_node_t(wayfire_toplevel_view view)
        : wf::scene::node_t(false), _view(view)
    {}

    ~pixdecor_shader_node_t() = default;

    wf::geometry_t get_bounding_box() override
    {
        if (!_view) return {0, 0, 0, 0};
        auto geo = _view->get_geometry();
        
        // Large shader background
        float scale = 2.0f;
        float offset_x = 100.0f;
        float offset_y = 100.0f;
        
        int extra_width = int(geo.width * (scale - 1.0f) / 2.0f);
        int extra_height = int(geo.height * (scale - 1.0f) / 2.0f);
        
        return wf::geometry_t{
            geo.x - extra_width + int(offset_x),
            geo.y - extra_height + int(offset_y),
            int(geo.width * scale),
            int(geo.height * scale)
        };
    }

    void gen_render_instances(std::vector<wf::scene::render_instance_uptr>& instances,
        wf::scene::damage_callback push_damage, wf::output_t *output = nullptr) override
    {
        instances.push_back(std::make_unique<shader_render_instance_t>(this, push_damage));
    }

    class shader_render_instance_t : public wf::scene::render_instance_t
    {
        pixdecor_shader_node_t *self;
        wf::scene::damage_callback push_damage;

    public:
        shader_render_instance_t(pixdecor_shader_node_t *s, wf::scene::damage_callback d)
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

            wf::auxilliary_buffer_t buffer;
            self->_view->take_snapshot(buffer);
            
            wlr_texture *raw_texture = buffer.get_texture();
            if (!raw_texture || (raw_texture->width == 0))
            {
                return;
            }

            wf::texture_t view_tex{raw_texture};
            wf::gles_texture_t gles_tex{view_tex};

            auto view_geo = self->_view->get_geometry();

            wf::gles::bind_render_buffer(data.target);

            GL_CALL(glEnable(GL_BLEND));
            GL_CALL(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

            // One large background shader layer
            float scale = 2.0f;
            float dx = 100.0f;
            float dy = 100.0f;
            
            float cx = (view_geo.width * (1.0f - scale)) / 2.0f;
            float cy = (view_geo.height * (1.0f - scale)) / 2.0f;
            
            wf::geometry_t shader_geo{
                int(view_geo.x + dx + cx),
                int(view_geo.y + dy + cy),
                int(view_geo.width * scale),
                int(view_geo.height * scale)
            };
            
            // Shader effect color
            glm::vec4 shader_tint(0.5f, 0.7f, 1.0f, 0.4f);

            wf::region_t shader_reg(shader_geo);
            auto shader_damage = data.damage & shader_reg;
            for (const auto& box : shader_damage)
            {
                wf::gles::render_target_logic_scissor(data.target, wlr_box_from_pixman_box(box));
                OpenGL::render_texture(gles_tex, data.target, shader_geo, shader_tint, 0);
            }

            GL_CALL(glDisable(GL_BLEND));
        }
    };
};
}  // namespace wf

#endif  // PIXDECOR_3D_SHADER_HPP