import {Menu, Icon} from "antd";
import React from 'react'
import { BrowserRouter as Router, Route, Link } from "react-router-dom";

const { SubMenu } = Menu;
const MenuItemGroup = Menu.ItemGroup;

export default (props) => {

    return (
        <Menu
            style={{height:"100vh"}}
            defaultOpenKeys={['sub1']}
            mode="inline"
        >
            <SubMenu key="sub1" title={<span><Icon type="mail" /><span>Features</span></span>}>

                    <Menu.Item key="/"><Link to={'/'}>Index</Link></Menu.Item>
                    <Menu.Item key="/report/"><Link to={'/tweets/'}>Tweets streaming</Link></Menu.Item>
                    <Menu.Item key="/map/"><Link to={'/map/'}>Vanity distribution</Link></Menu.Item>
                    <Menu.Item key="/experiment/"><Link to={'/experiment/'}>Face lab</Link></Menu.Item>

            </SubMenu>

        </Menu>
        )

}
